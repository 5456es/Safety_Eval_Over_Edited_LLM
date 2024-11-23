#!/bin/bash

cd ../../experiment/ROME/

# 打印当前目录
pwd

# 定义数据大小数组（注意不要使用逗号分隔�?
data_sizes=(1)

# 遍历数据大小数组
for size in "${data_sizes[@]}"; do
    python mistral-7b-instruct-v0.3_edit.py \
    --ds_size "$size" \
    --hparams_dir ../../src/hparams/ROME/mistral-7b-instruct-v0.3-kenji.yaml
done
