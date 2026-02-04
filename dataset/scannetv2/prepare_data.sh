#!/bin/bash
echo Copying Data
python3 split_data.py --data_path ../../../../dataset
echo Preprocessing Data
python3 prepare_data_inst.py
python3 prepare_superpoint.py
