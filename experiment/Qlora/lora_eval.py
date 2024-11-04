import os
import json
import argparse
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType,PeftModel
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', '..', 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from eval_utils import llama_safety_eval,mistral_safety_eval


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora_path',type=str,required=True)
    parser.add_argument('--safety_eval_data',type=str,default=
                        '../../data/eval_data/merged_data_2024-11-04.json'
                        )
    parser.add_argument('--model',required=True,type=str)
    parser.add_argument('--eval_data_size',type=int,default=-1)
    parser.add_argument('--data_source',type=str,required=True)
    parser.add_argument('--data_size',type=str,required=True)
    parser.add_argument('--safety_eval_output',type=str,required=True)

    args=parser.parse_args()


    
    # 加载预训练模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model,device_map='auto')

    print('We are now merging the lora adapter to base model')
    lora_model = PeftModel.from_pretrained(model, args.lora_path)
    merged_model = lora_model.merge_and_unload()

    print("-"*50)
    print("\n"*10)
    print("Now we start evaluating")

    tag=str(args.data_source)+'_'+str(args.data_size)
    from datetime import datetime

    current_time = datetime.now().strftime("%m%d_%H%M")
    eval_save_dir=os.path.join(args.safety_eval_output)

    ### load tokenizer according to the hparams
    for eval_data_source in ['adv_train', 'GCG', 'mix_eval_freeform_0811']:
        if 'mistral' in args.model:
            mistral_safety_eval(merged_model,
                    args.model,
                    args.safety_eval_data,
                    eval_data_source,
                    args.eval_data_size,
                    eval_save_dir)
        elif 'llama' in args.model:
            llama_safety_eval(merged_model,
                    args.model,
                    args.safety_eval_data,
                    eval_data_source,
                    args.eval_data_size,
                    eval_save_dir)





