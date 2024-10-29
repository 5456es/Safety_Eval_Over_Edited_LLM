#!/bin/bash

#SBATCH --job-name=edit_llama_rome      # task name
#SBATCH --gpus=a100-80:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --time=0-24:30:00          # 设置作业的最大运行时间为2小时30分钟

# 激活conda环境
source ~/miniconda3/bin/activate ee  # 使用你安装的conda环境


cd ../experiment/MEMIT/
pwd

python llama2-7b-hf-chat_edit.py \
	--hparams_dir ../../src/hparams/MEMIT/llama2-7b-hf-chat-cluster.yaml
