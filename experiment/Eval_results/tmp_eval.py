import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

models=['llama2-7b-hf-chat','mistral-7b-instruct-v0.3']
methods=['lora']
eval_sets=['adv_train','GCG','mix_eval_freeform_0811']
part_num=3

with open('../../src/rejection_list.json','r') as f:
    rejection_list=json.load(f)

with open('../../data/eval_data/unified_mixeval_0811_freeform_shortend_no_context_inuse.json','r') as f:
    mix_eval_data=json.load(f)
    prompts_targets={entry['prompt']:entry['target'] for entry in mix_eval_data}



def rejection_list_check(text):
    if any(word.lower() in text.lower() for word in rejection_list):
        return 1
    else:
        return 0

def mix_eval_check(input,output):
    for prompt in prompts_targets.keys():
        if prompt in input:
            target=prompts_targets[prompt]
            for item in target:
                if item.lower() in output.lower():
                    return 1
            
            return 0
    return -1
    
def rejection_eval(results):
    rejection_results=[]
    for result in results:
        rejection_result=rejection_list_check(result['output'].split('/INST')[1])
        rejection_results.append(rejection_result)
        result['rejection_result']=rejection_result

    
    # 计算 'rejec' 和 'total'
    rejec_count = rejection_results.count(1)
    total_count = len(rejection_results)
    return {'rejec': rejec_count, 'total': total_count,'performance':rejec_count/total_count}

def mix_eval(results):
    eval_results=[]
    for result in results:
        output=result['output'].split('/INST')[1]
        input=result['input']
        eval_result=mix_eval_check(input,output)
        eval_results.append(eval_result)
        result['eval_result']=eval_result
    correct_count=eval_results.count(1)
    total_count=len(eval_results)-eval_results.count(-1)

    return {'correct': correct_count, 'total': total_count,'performance':correct_count/total_count}
        

def analyse_model_method_edittimes(path):
    analysis={eval_set:[] for eval_set in eval_sets}
    for part in os.listdir(path):
        try:
            for eval_set in eval_sets:
            
                with open(os.path.join(path,part,'eval',eval_set,'results.json'),'r') as f:
                    results=f.readlines()[1:]
                    results=[json.loads(line) for line in results]
                    if  'mix' in eval_set:
                        analysis_=mix_eval(results)
                    else:
                        analysis_=rejection_eval(results)
                    analysis[eval_set].append(analysis_)
        except:
            print(os.path.join(path,part))
    return analysis

paths=[]

for model in models:
    for method in methods:
        paths.append(os.path.join('../../results',method,model))
analysis={}
for model_method_path in tqdm(paths,total=len(paths)):
    for edit_time in os.listdir(model_method_path):
        
        name=edit_time
        model_method_edittimes_path=os.path.join(model_method_path,edit_time)
        analysis[model_method_edittimes_path]=analyse_model_method_edittimes(model_method_edittimes_path)
   


sorted_analysis = {key: analysis[key] for key in sorted(analysis.keys())}


for model_method_times in analysis.keys():
    for eval_set in analysis[model_method_times].keys():
        performance=[entry['performance'] for entry in analysis[model_method_times][eval_set]]
        mean_performance = sum(performance) / len(performance)
        analysis[model_method_times][eval_set].append({'avg_performance':mean_performance})

with open('analysis.json','w') as f:
    json.dump(sorted_analysis,f,indent=4)
    






        



# Plotting the avg_performance for each model across different benchmarks
for model in models:
    for eval_set in eval_sets:
        x_edit_times = []
        y_avg_performance = []
        
        for model_method_times, data in sorted_analysis.items():
            if model in model_method_times:
                edit_time = int((model_method_times.split('/')[-1]).split('_')[-1])  # Assuming edit_time is represented as an integer in folder names
                x_edit_times.append(edit_time)
                avg_performance = [entry['avg_performance'] for entry in data[eval_set] if 'avg_performance' in entry]
                if avg_performance:
                    y_avg_performance.append(avg_performance[0])

        # Sort by edit times to ensure correct order in plotting
        sorted_data = sorted(zip(x_edit_times, y_avg_performance))
        x_edit_times, y_avg_performance = zip(*sorted_data)

        # Plotting
        plt.figure()
        plt.plot(x_edit_times, y_avg_performance, marker='o')
        plt.xlabel('Edit Times')
        plt.ylabel('Avg Performance')
        plt.ylim(0, 1.2)  # Set y-axis range to 0-1

        plt.title(f'{model} - {eval_set} Performance Across Edit Times')
        plt.grid(True)

        # Save the plot
        plot_filename = f'{model}_{eval_set}_performance.png'
        plt.savefig(plot_filename)
        plt.close()