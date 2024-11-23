#!/bin/bash

# 进入目标目录
cd ../../experiment/ROME

# 打印当前工作目录
pwd

data_sizes=(1)

# 遍历数据大小数组，并将其传递给 Python 脚本
for size in "${data_sizes[@]}"; do
    python gpt2-xl_edit.py \
    --ds_size "$size" \
    --hparams_dir ../../src/hparams/ROME/gpt2-xl-kenji.yaml
done
