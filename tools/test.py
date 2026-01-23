import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from munch import Munch
from torch.nn.parallel import DistributedDataParallel
from pathlib import Path
import argparse
import datetime
import os
import os.path as osp
import shutil
import time
from spherical_mask.data import build_dataloader, build_dataset
from spherical_mask.evaluation import PointWiseEval, ScanNetEval
from spherical_mask.model import SphericalMask
from spherical_mask.model.criterion_spherical_mask import Criterion_SphericalMask
from spherical_mask.util import (
    AverageMeter,
    SummaryWriter,
    build_optimizer,
    get_dist_info,
    get_root_logger,
    init_dist,
    load_checkpoint
)

np.random.seed(0)
torch.manual_seed(0)

SEMANTIC_IDXS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])


def get_args():
    parser = argparse.ArgumentParser("ISBNet")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("--dist", action="store_true", help="run with distributed parallel")
    parser.add_argument("--resume", type=str, help="path to resume from")
    parser.add_argument("--work_dir", type=str, help="working directory")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--out", type=str, default=None, help="Directory to save predictions. If not provided, predictions will NOT be saved.")

    args = parser.parse_args()
    return args


def test(model, val_loader, cfg, logger):
    logger.info("Test")
    all_pred_insts, all_sem_labels, all_ins_labels = [], [], []

    val_set = val_loader.dataset

    point_eval = PointWiseEval(num_classes=cfg.model.semantic_classes)
    scannet_eval = ScanNetEval(val_set.CLASSES, dataset_name=cfg.data.train.type)

    save_outputs = cfg.out is not None
    if save_outputs:
        pred_save_dir = osp.join(cfg.out, "pred_instance")
        os.makedirs(pred_save_dir, exist_ok=True)
        logger.info(f"Saving predictions to: {pred_save_dir}")
    else:
        logger.info("No --out provided => outputs will NOT be saved.")

    torch.cuda.empty_cache()

    model.iterative_sampling = False
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(val_loader):
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                res = model(batch)

            logger.info(f"Infer scene {i+1}/{len(val_set)}")

            if cfg.model.semantic_only:
                point_eval.update(
                    res["semantic_preds"],
                    res["centroid_offset"],
                    res["corners_offset"],
                    res["semantic_labels"],
                    res["centroid_offset_labels"],
                    res["corners_offset_labels"],
                    res["instance_labels"],
                )
            else:
                pred_instances_list = res["pred_instances"]

                for inst_dict in pred_instances_list:
                    inst_dict['pred_mask'] = inst_dict['pred_mask'].cpu()

                all_pred_insts.append(pred_instances_list)
                all_sem_labels.append(res["semantic_labels"])
                all_ins_labels.append(res["instance_labels"])

                # Scene name
                try:
                    scene_name = val_set.filenames[i]
                    scene_name = Path(scene_name).stem.replace('_inst_nostuff', '')
                except Exception as e:
                    logger.error(f"Could not get scene name. Error: {e}")
                    continue

                # Only SAVE if --out exists
                if save_outputs:
                    summary_file_path = osp.join(pred_save_dir, f"{scene_name}.txt")

                    try:
                        label_shift = cfg.model.semantic_classes - cfg.model.instance_classes
                    except Exception:
                        logger.warning("Could not find label_shift; defaulting to 2.")
                        label_shift = 2

                    with open(summary_file_path, "w") as f:
                        for inst_idx, inst in enumerate(pred_instances_list):
                            pred_mask_tensor = inst['pred_mask']
                            score = inst['conf']

                            model_label_id = inst['label_id']
                            try:
                                remapped_idx = int(model_label_id) + label_shift - 1
                                viz_label_id = SEMANTIC_IDXS[remapped_idx]
                            except Exception:
                                viz_label_id = model_label_id

                            mask_file_name = f"{scene_name}_{inst_idx:03d}.txt"
                            mask_file_path = osp.join(pred_save_dir, mask_file_name)

                            try:
                                mask_data_np = pred_mask_tensor.numpy().astype(np.uint8)
                                np.savetxt(mask_file_path, mask_data_np, fmt="%d")
                            except Exception as e:
                                logger.error(f"Failed to save mask {mask_file_path}. Error: {e}")
                                continue

                            f.write(f"{mask_file_name} {viz_label_id} {score}\n")

                    logger.info(f"Saved prediction files for scene: {scene_name}")

    if cfg.model.semantic_only:
        logger.info("Evaluate semantic segmentation and offset MAE")
        miou, acc, mae = point_eval.get_eval(logger)
        return miou
    else:
        logger.info("Evaluate instance segmentation")
        eval_res = scannet_eval.evaluate(all_pred_insts, all_sem_labels, all_ins_labels)
        del all_pred_insts, all_sem_labels, all_ins_labels

    return eval_res['all_ap']


def main():
    args = get_args()
    cfg_txt = open(args.config, "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))

    cfg.out = args.out

    if args.dist:
        init_dist()
    cfg.dist = args.dist

    # work_dir & logger
    if os.path.exists(cfg.work_dir) == False:
        os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    logger.info(f"Config:\n{cfg_txt}")
    logger.info(f"Distributed: {args.dist}")
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))

    # model
    criterion = Criterion_SphericalMask(
        cfg.model.semantic_classes,
        cfg.model.instance_classes,
        cfg.model.semantic_weight,
        cfg.model.ignore_label,
        semantic_only=cfg.model.semantic_only,
        total_epoch=cfg.epochs,
        voxel_scale=cfg.data.train.voxel_cfg.scale,
    )

    model = SphericalMask(**cfg.model, criterion=criterion, dataset_name=cfg.data.train.type, trainall=False).cuda()

    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    logger.info(f"Total params: {total_params}")
    logger.info(f"Trainable params: {trainable_params}")

    if args.dist:
        model = DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()],
            find_unused_parameters=(trainable_params < total_params)
        )

    cfg.data.test['debug'] = False
    val_set = build_dataset(cfg.data.test, logger)

    test_loader = build_dataloader(val_set, training=False, dist=False, **cfg.dataloader.test)

    # load_model
    assert args.ckpt is not None, 'ckpt path must be provided for testing.'
    assert os.path.exists(args.ckpt), f"{args.ckpt} does not exist."
    load_checkpoint(args.ckpt, logger, model)

    # test
    torch.cuda.empty_cache()
    ap = test(model, test_loader, cfg, logger)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()