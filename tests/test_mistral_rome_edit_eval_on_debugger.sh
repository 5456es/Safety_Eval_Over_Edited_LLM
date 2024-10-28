#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mistral

cd ../experiment/ROME/
pwd

python mistral-7b-instruct-v0.3_edit.py

