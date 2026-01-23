"""Modified from SparseConvNet data preparation: https://github.com/facebookres
earch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py."""

import numpy as np
import plyfile
import scannet_util
import torch
import glob
import json
import multiprocessing as mp

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i

files = sorted(glob.glob("test/*.ply"))

def f(fn):
    print(fn)

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    torch.save((coords, colors), fn[:-4] + "_inst_nostuff.pth")
    print("Saving to " + fn[:-4] + "_inst_nostuff.pth")

p = mp.Pool(processes=mp.cpu_count())
p.map(f, files)
p.close()
p.join()