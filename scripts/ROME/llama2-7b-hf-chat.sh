#!/bin/bash

# 进入目标目录
cd ../../experiment/Qlora

# 打印当前工作目录
pwd

# 定义一个数组，包含你想要的参数值
data_sizes=(1 5 10 15 20 25 30 35 40 45 50)

# 遍历数据大小数组，并将其传递给 Python 脚本
for size in "${data_sizes[@]}"; do
    python llama2-7b-hf-chat_edit.py --ds_size "$size" --hparams_dir ../../src/hparams/ROME/llama2-7b-hf-chat-cluster.yaml
done
