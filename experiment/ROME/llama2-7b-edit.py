import os.path
import sys
import json
import argparse
import shutil
from easyeditor import ROMEHyperParams, KnowEditDataset, BaseEditor
from utils import prepare_knowedit_data

# Set up the directory for importing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', '..', 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument("--hparams_dir", default='../../src/hparams/ROME/llama2-7b-hf-chat-debugger.yaml', type=str)
    parser.add_argument("--sequential_edit", default=True, type=bool)
    parser.add_argument("--data_path", default="../../edit_data/edit_data/KnowEdit/benchmark/ZsRE/zsre_mend_eval_portability_gpt4.json", type=str)
    parser.add_argument("--data_source", default="ZsRE", type=str)
    parser.add_argument("--ds_size", default=1, type=int)
    parser.add_argument("--random", default=False, type=bool)
    parser.add_argument("--id_start", default=0, type=int)
    parser.add_argument("--results_save_dir", default="../../logs/ROME/", type=str)
    parser.add_argument("--safety_eval_output", default="../../results/ROME/", type=str)

    args = parser.parse_args()

    # Load data with error handling
    try:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"Successfully loaded data from {args.data_path}")
    except FileNotFoundError:
        print(f"Error: Data file {args.data_path} not found.")
        sys.exit(1)

    # Dataset preparation
    dataset = KnowEditDataset(args.data_path, source=args.data_source, size=args.ds_size, id_start=args.id_start)
    prompts, subjects, target_new, _, _ = prepare_knowedit_data(dataset)

    # Load hyperparameters and create editor
    editing_hparams = ROMEHyperParams
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    editor = BaseEditor.from_hparams(hparams)

    # Define prompts and rephrased prompts if necessary
    prompts = [test_data_['src'] for test_data_ in test_data]
    rephrase_prompts = [edit_data['rephrase'] for edit_data in dataset]

    # Perform the edit
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        subject=subjects,
        keep_original_weight=False,
        sequential_edit=args.sequential_edit
    )

    # Saving results
    from datetime import datetime
    current_time = datetime.now().strftime("%m%d_%H%M")
    model_name = "llama2-7b-hf-chat"

    data_source, data_size = args.data_source, args.ds_size
    tag = f"{data_source}_{data_size}"
    start_id = f"{args.id_start}"

    save_dir = os.path.join(args.results_save_dir, model_name, tag, start_id)
    eval_save_dir = os.path.join(args.safety_eval_output, model_name, tag, start_id)
    os.makedirs(save_dir, exist_ok=True)

    metrics_save_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    args_save_path = os.path.join(save_dir, "args.json")
    with open(args_save_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    # Move logs folder with error check
    logs_dir = "./logs"
    destination_dir = os.path.join(save_dir, "logs")

    if os.path.exists(logs_dir):
        shutil.move(logs_dir, destination_dir)
        print(f"Moved logs to {destination_dir}")
    else:
        print(f"Directory {logs_dir} does not exist.")