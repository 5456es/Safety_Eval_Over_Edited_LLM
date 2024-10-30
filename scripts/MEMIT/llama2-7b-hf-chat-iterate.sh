#!/bin/bash

#SBATCH --job-name=edit_llama_rome      # task name
#SBATCH --gpus=a100-80:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --time=0-24:30:00          # 设置作业的最大运行时间为2小时30分钟

# 进入目标目录
source ~/miniconda3/bin/activate ee  # 使用你安装的conda环境

cd ../../experiment/MEMIT
#!/bin/bash

# 打印当前工作目录
pwd

# 定义数据大小数组
data_sizes=(1 )

# 遍历数据大小数组并递增 id_start
id_start=0
for size in "${data_sizes[@]}"; do
    while [ "$id_start" -le 1300 ]; do
        python llama2-7b-hf-chat_edit.py --ds_size "$size" \
            --hparams_dir ../../src/hparams/MEMIT/llama2-7b-hf-chat-cluster.yaml \
            --id_start "$id_start"
        
        # 增加 id_start 的值
        ((id_start++))
    done
done
