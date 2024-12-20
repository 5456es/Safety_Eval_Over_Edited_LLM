#!/bin/bash

#SBATCH --job-name=edit_mistral_lora      # task name
#SBATCH --gpus=a100-80:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --time=0-24:30:00          # 设置作业的最大运行时间为2小时30分钟

# 激活conda环境
source ~/miniconda3/bin/activate ee  # 使用你安装的conda环境

cd  ../../experiment/Qlora

pwd

data_parts=(0 1 2)
data_source=ZsRE
data_sizes=(1 5  10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95)
data_sizes=(100)
for data_part in "${data_parts[@]}"; do
    for data_size in "${data_sizes[@]}"; do
        bash train_mistral-7b-instruct-v0.3_wo_tmp.sh "$data_part" "$data_source" "$data_size"
    done
done

# data_part , data_source , data_size



