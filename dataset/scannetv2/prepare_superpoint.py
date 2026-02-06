import numpy as np
import torch
import trimesh
import os
import segmentator
import glob


def get_superpoint(mesh_file):
    mesh = trimesh.load(mesh_file, force='mesh', process=False)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.faces).astype(np.int64))
    superpoint = segmentator.segment_mesh(vertices, faces)
    return superpoint


if __name__ == "__main__":
    os.makedirs("superpoints", exist_ok=True)
    scans = sorted(glob.glob("test/*.ply"))
    for scan in scans:
        spp = get_superpoint(scan)
        spp = spp.numpy()
        scan_name = scan.split("/")[-1].split(".")[0]
        torch.save(spp, os.path.join("superpoints", "{}.pth".format(scan_name)))
