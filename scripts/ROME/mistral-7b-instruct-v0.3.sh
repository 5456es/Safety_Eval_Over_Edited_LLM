#!/bin/bash

cd ../../experiment/ROME/mistral-7b-instruct-v0.3_edit.py

# 打印当前目录
pwd

# 定义数据大小数组（注意不要使用逗号分隔�?
data_sizes=(1 5 10 15 20 25 30 35 40 45 50)

# 遍历数据大小数组
for size in "${data_sizes[@]}"; do
    python mistral-7b-instruct-v0.3_edit.py --ds_size "$size"
done
