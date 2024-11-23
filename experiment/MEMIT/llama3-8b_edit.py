import os.path
import sys
import time
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', '..', 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

import json
import random
import argparse
import shutil
from easyeditor import MEMITHyperParams
from easyeditor import KnowEditDataset
from easyeditor import BaseEditor
from utils import prepare_knowedit_data
from eval_utils import llama_safety_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Parameters
    parser.add_argument("--hparams_dir", default='../../src/hparams/MEMIT/llama2-7b-hf-chat-debugger.yaml',type=str)
    ### Default is Sequential Edit
    parser.add_argument("--sequential_edit", default=True, type=bool)
    ## Data
    parser.add_argument(
        "--data_path", default="../../data/edit_data/merged_data.json", type=str
    )
    ### type of data
    #### ZsRE,wiki_recent,wiki_counterfact,NEWS2024,Mixed
    parser.add_argument("--data_source", default="ZsRE", type=str)
    ### size of the dataset
    parser.add_argument("--ds_size", default=1, type=int)
    ### whether random
    parser.add_argument("--random",default=False,type=bool)
    parser.add_argument("--id_start", default=0,type=int)


    ## Output and logging
    ### results save directory
    parser.add_argument("--results_save_dir", default="../../logs/MEMIT/", type=str)



    # Eval data path
    parser.add_argument("--safety_eval_data",type=str,default="../../data/eval_data/merged_data_2024-10-18.json")
    # Eval data num
    parser.add_argument("--eval_data_size",default=-1,type=int)
    ### Eval results save path
    parser.add_argument("--safty_eval_output",default="../../results/MEMIT/",type=str)
    
    args = parser.parse_args()

    print(f"Loading data from {args.data_path}")
    dataset = KnowEditDataset(
        args.data_path, source=args.data_source, size=args.ds_size,id_start=args.id_start
    )
    prompts, subjects, target_new, _, _ = prepare_knowedit_data(dataset)

    print(f"Prepare for params from {args.hparams_dir}")
    editing_hparams = MEMITHyperParams
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    editor = BaseEditor.from_hparams(hparams)

    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=target_new,
        subject=subjects,
        keep_original_weight=False,
        sequential_edit=True,
    )

    torch.cuda.empty_cache()
    from datetime import datetime

    current_time = datetime.now().strftime("%m%d_%H%M")
    model_name = "llama3-8b"

    data_source,data_size=args.data_source,args.ds_size
    tag=str(data_source)+'_'+str(data_size)
    start_id=f"{args.id_start}"


    save_dir = os.path.join(args.results_save_dir, model_name, tag,start_id)
    eval_save_dir=os.path.join(args.safty_eval_output, model_name, tag,start_id)
    os.makedirs(save_dir, exist_ok=True)

    # 淇濆瓨 metrics 鍒版寚瀹氭枃浠跺す涓�
    metrics_save_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # 淇濆瓨 args 鍒版寚瀹氭枃浠跺す涓�
    args_save_path = os.path.join(save_dir, "args.json")
    with open(args_save_path, "w") as f:
        json.dump(vars(args), f, indent=4)


    # 鎶婂綋鍓嶇洰褰曚笅鍙仛 logs 鐨勭洰褰曠Щ鍔ㄥ埌 save_dir
    logs_dir = "./logs"  # 褰撳墠鐩綍涓嬬殑 logs 鏂囦欢澶�
    destination_dir = os.path.join(save_dir, "logs")  # 鐩爣鐩綍涓殑 logs 鏂囦欢澶�

    # 绉诲姩 logs 鏂囦欢澶�
    if os.path.exists(logs_dir):
        shutil.move(logs_dir, destination_dir)
    else:
        print(f"Directory {logs_dir} does not exist.")



    # print("-"*50)
    # print("\n"*10)
    # print("Now we start evaluating")

    # ### load tokenizer according to the hparams
    # for eval_data_source in ['adv_train', 'GCG', 'mix_eval_freeform_0811']:
    #     llama_safety_eval(edited_model,
    #             hparams.model_name,
    #             args.safety_eval_data,
    #             eval_data_source,
    #             args.eval_data_size,
    #             eval_save_dir)
