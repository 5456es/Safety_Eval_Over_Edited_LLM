import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType,PeftModel


prompt=["[INST] how do you feel about the day? [/INST]"]


# 加载预训练模型和tokenizer
model_name = "../../.hf_cache/mistral-7b-instruct-v0.3"
model = AutoModelForCausalLM.from_pretrained(model_name,device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer(prompt, return_tensors="pt")  # 将输入转为张量并放到 CUDA 设备上

# 生成输出
output_ids = model.generate(inputs["input_ids"], max_length=50)  # 指定生成的最大长度
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)  # 解码输出，忽略特殊标记

print(output_text)

lora_model = PeftModel.from_pretrained(model, "./out/mistral_instruction_mistral")
merged_model = lora_model.merge_and_unload()

inputs = tokenizer(prompt, return_tensors="pt")  # 将输入转为张量并放到 CUDA 设备上

# 生成输出
output_ids = merged_model.generate(inputs["input_ids"], max_length=50)  # 指定生成的最大长度
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)  # 解码输出，忽略特殊标记

print(output_text)



