import transformers
import argparse
import torch
import ast
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import json
import pandas as pd
import os

# Method 1 - Free form text
# Method 2 - Supressing tokens

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:{instruction}\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task."
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:{instruction}\n\n### Response:"
    ),
    "prompt_input_task": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Task:{task}\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input_task": (
        "Below is an instruction that describes a task."
        "Write a response that appropriately completes the request.\n\n"
        "### Task:\n{task}\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
# we need to add the values to the tokenizer's vocab to finetune
TASK_DICT = {
    "t2t":"Text2Text",
    "t2i":"Text2Image",
    "i2t":"Image2Text",
    "s2t":"Speech2Text",
    "t2s":"Text2Speech"
}


def main(args):
    
    device = args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"


    # Load model and tokenizer
    # model_name = "GLAM24/phi2_baseline_240604_glam_instruct_1m"
    
    # model_name = "GLAM24/GVLAM-Llama-3.1-8B-instruct-1m"
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map=device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Generate text
    max_length = 512
    num_return_sequences = 1

    
    # prompt_format = prompt.format(instruction=instruction, options=options)
    
    configs = ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']
    
    # configs = ['Accounting']
    
    # Generation kwargs
    gen_kwargs = {
        "num_beams": 1,
        "output_scores": True,
        "output_logits": True,
        "return_dict_in_generate": True,
    }
    
    def get_probs(model, inputs, label):
        
        # Process the label to generate output_ids for comparison
        output_ids = tokenizer(str(label), return_tensors="pt", add_special_tokens=False)['input_ids'].to(device)
        
        # import pdb;pdb.set_trace()
        # Generate model outputs
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=output_ids.shape[-1], 
                min_new_tokens=output_ids.shape[-1], 
                num_beams=1,
                output_scores=True,
                output_logits=True,
                return_dict_in_generate=True
            )

        logits = torch.stack(output.logits, dim=1)
        
        # Calculate the probabilities for all tokens in the label
        probs = torch.gather(F.softmax(logits, dim=2), 2, output_ids.unsqueeze(2).to(device))
        return probs.sum().detach().cpu().item()
        
    
    
    for config in configs:
        
        # download the dataset
        dataset = datasets.load_dataset("MMMU/MMMU", config, split="validation")
        
        image_tokenized = pd.read_csv(f"MMMUTokenized/{config}.csv")
    
        predictions = []  
        options_all = []
        
        # create a directory for each model to save results
        root_dir = f"{args.output_dir}/{args.model_name_or_path.split('/')[-1]}"
        os.makedirs(root_dir, exist_ok=True)
                
        # iterate over all examples in the test
        for idx in tqdm(range(len(dataset)), desc=f"Evaluation {args.model_name_or_path.split('/')[-1]} for {config}: "):
        
            
            example = dataset[idx]
            
            # instruction, options
            options = " ".join([f"{chr(65 + i)}. {label}" for i, label in enumerate(ast.literal_eval(example['options']))])
           
            # import pdb;pdb.set_trace()
            instruction = example['question'] + f" Options: {options} Answer:"
            image_tokens = image_tokenized.loc[idx]['image']
            
            prompt_format = PROMPT_DICT['prompt_input_task'].format(task=TASK_DICT['i2t'], instruction=instruction, input=image_tokens)
               
            inputs = tokenizer(prompt_format, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}  # Move each tensor to the device
                
            prompt_length = inputs['input_ids'].shape[1]
            
            
            # get the probablities of each class
            # labels = [f"{chr(65 + i)}" for i, label in enumerate(ast.literal_eval(example['options']))]
            # probs = [get_probs(model, inputs, label) for label in [f"{chr(65 + i)}" for i, label in enumerate(ast.literal_eval(example['options']))]]
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,                
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.eos_token_id,  # To avoid warnings for models without pad_token_id
                temperature=0.0,
            )
            answer = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
            
            predictions.append(answer)
            options_all.append(options)
        
        # save the predictions into csv files
        pd.DataFrame(data={"question":dataset['question'], "options":options_all, "answer":dataset['answer'], "generated_answer":predictions}).to_csv(f"{root_dir}/{config}.csv", index=False)

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(prog='eval_mmmu.py',description='zero-shot glam-inference',epilog='Text at the bottom of help')
    parser.add_argument("--model_name_or_path", type=str, default="GLAM24/phi2_baseline_240604_glam_instruct_1m", required=False, help="Provide model name here.",)
    parser.add_argument("--dataset_name_or_path", type=str, default="MMMU/MMMU", required=False)
    parser.add_argument("--task_name", type=str, default="mmmu", required=False)
    parser.add_argument("--split_name", type=str, default="validation", required=False)
    parser.add_argument("--device", type=str, default=None, required=False)
    parser.add_argument("--output_dir", type=str, default="MMMUGenResults/",)
    
    args = parser.parse_args()
    main(args=args)