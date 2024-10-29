#!/bin/bash

# 进入目标目录
cd ../../experiment/MEMIT

# 打印当前工作目录
pwd

# 定义数据大小数组
data_sizes=(1 5 10 15 20 25 30 35 40 45 50)

# 遍历数据大小数组，并将每个数据大小传递给 Python 脚本
for size in "${data_sizes[@]}"; do
    python llama2-7b-hf-chat_edit.py --ds_size "$size" --hparams_dir ../../src/hparams/MEMIT/llama2-7b-hf-chat-cluster.yaml
done
