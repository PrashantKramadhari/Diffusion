import os
import sys
from pathlib import Path
import json
import subprocess
import time

def force_cleanup():
    """Force cleanup of GPU memory and processes"""
    # Kill any remaining python processes
    try:
        subprocess.run(['pkill', '-f', 'main.py'], check=False)
    except:
        pass
    
    # Clear GPU memory if possible
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass
    
    # Wait a bit for cleanup
    time.sleep(2)

def auto_run_eval(json_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        prompts_data = json.load(f)
    
    # Now prompts_data contains the JSON object
    print(f"Loaded {len(prompts_data)} prompts from JSON file")
    

    if "wide_image" in json_file_path:
        app = "wide_image"
        for key, value in prompts_data.items():
            print('Running: ', key)
            prompt = value['prompt']
            eval_pos = value.get('eval_pos')
            eval_pos_str = ' '.join(map(str, eval_pos))
            command = f"python main.py --app {app} --prompt \"{prompt}\" --tag wide_image --save_dir_now --eval_pos {eval_pos_str} --save_top_dir ./eval_outputs"
            result = os.system(command)
            
            # Force cleanup between runs
            force_cleanup()
            
            if result == 0:
                print(f"✓ Completed: {key}")
            else:
                print(f"✗ Failed: {key}")
    elif "ambiguous_image" in json_file_path:
        app = "ambiguous_image"
        for key, value in prompts_data.items():
            print('Running: ', key)
            canonical_prompt = value.get('canonical_prompt')
            instance_prompt = value.get('instance_prompt')
            # python main.py --app ambiguous_image --prompts 'an oil painting of a cat' 'a oil painting of a truck' --tag ambiguous_image --save_dir_now
            command = f"python main.py --app ambiguous_image --prompts '{canonical_prompt}' '{instance_prompt}' --tag ambiguous_image --save_dir_now --save_top_dir ./eval_outputs"
            result = os.system(command)
            
            # Force cleanup between runs
            force_cleanup()
            
            if result == 0:
                print(f"✓ Completed: {key}")
            else:
                print(f"✗ Failed: {key}")



if __name__ == "__main__":
    auto_run_eval("data/wide_image_prompts.json")
    # auto_run_eval("data/ambiguous_image_prompts.json")