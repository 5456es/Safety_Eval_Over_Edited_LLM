
base_dir=../../results/lora/mistral-7b-instruct-v0.3
model=../../.hf_cache/mistral-7b-instruct-v0.3
model_id=mistral-7b-instruct-v0.3
conv_template=mistral
instruction_type=instruction # natural_instruction, unnatural_instruction, random_instruction, no_instruction


# 从命令行参数获取变量值
data_part=${1:-0}       # 默认值为0
data_source=${2:-"ZsRE"}  # 默认值为ZsRE，可以是ZsRE, Wiki_recent, Wiki_counterfact
data_size=${3:-1}       # 默认值为1


wandb online

exp_name=${data_source}_${data_size}
output_dir=$base_dir/${exp_name}/part_${data_part}
mkdir -p $output_dir


python prepare_edit_data_for_lora.py \
  --data_part ${data_part} \
  --data_size ${data_size} \
  --data_source ${data_source} \

rm -rf "${output_dir}/wandb"
rm "${output_dir}/tmp_data.jsonl"

python train_lora_wo_tmp.py \
  --data-path ./tmp_data.jsonl \
  --output_dir $output_dir \
  --wandb_run_name ${model_id}_${exp_name} \
  --base_model $model \
  --batch_size=$(( data_size < 32 ? data_size : 32 )) \
  --micro_batch_size 1 \
  --learning_rate 0.0004 \
  --cutoff_len 4096 \
  --val_set_size 0 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules '[gate_proj, down_proj, up_proj]' \
  --train_on_inputs False \
  --add_eos_token True \
  --group_by_length False \
  --lr_scheduler 'cosine' \
  --warmup_steps 100 \
  --wandb_project llm-edit\
  --num_epochs 10 \
  --conv_template $conv_template \
  --prompt_format instruction \
  --use_cot False \
  --instruction_type $instruction_type \
  --output_type output \
  2>&1 | tee $output_dir/log.txt


mv ./tmp_data.jsonl ${output_dir}
mv  ./wandb ${output_dir}



## eval over several benchmarks
python lora_eval.py \
  --lora_path $output_dir \
  --model ${model} \
  --data_source ${data_source} \
  --data_size ${data_size} \
  --safety_eval_output ${output_dir}/eval

## remove lora model
python lora_clean.py \
  --clean_path $output_dir


# # mixeval inference
# python -m mix_eval.evaluate \
#   --model_name local_chat \
#   --model_path ${output_dir} \
#   --data_path ./MixEval/mix_eval/data \
#   --benchmark mixeval \
#   --version 2024-08-11 \
#   --batch_size 16 \
#   --output_dir ./data/bench/mixeval/out/${model_id} \
#   --api_parallel_num 20 \
#   --conv_template $conv_template \
#   --inference_only

# # alpaca_eval inference
# python -m utils.eval.generate default \
#   --bench alpaca_eval \
#   --model_path $model \
#   --adapter_model_path ${output_dir} \
#   --conv_template $conv_template \
#   --model_id $model_id \
#   --batch_size 16 \
#   --max_new_len 512 \
#   --output_file_format json
