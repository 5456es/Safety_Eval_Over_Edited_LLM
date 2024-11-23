#!/bin/bash

# 进入目标目录
cd ../../experiment/GRACE

# 打印当前工作目录
pwd

# 定义数据大小数组
data_sizes=(1)

# 遍历数据大小数组，并将每个数据大小传递给 Python 脚本
for size in "${data_sizes[@]}"; do
    python llama3-8b_edit.py \
    --ds_size "$size" \
    --hparams_dir ../../src/hparams/GRACE/llama3-8b-kenji.yaml
done
