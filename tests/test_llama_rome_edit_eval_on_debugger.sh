#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mistral

cd ../experiment/ROME/
pwd

python llama2-7b-hf-chat_edit.py 

