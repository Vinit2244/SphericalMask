import os
import shutil
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", help="path to dataset folder")
opt = parser.parse_args()

scans = sorted(glob.glob(opt.data_path + "/*.ply"))
os.makedirs("test", exist_ok=True)
for src in scans:
    scan = src.split("/")[-1].strip()
    dest = "test/{}".format(scan)
    shutil.copyfile(src, dest)
print("done")
