#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mistral

cd ../experiment/MEMIT/
pwd

python llama2-7b-hf-chat_edit.py   \
	--hparams_dir ../../src/hparams/MEMIT/llama2-7b-hf-chat-cluster.yaml


