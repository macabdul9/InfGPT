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
import random

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
    # create a directory for each model to save results
    root_dir = f"{args.output_dir}/{args.model_name_or_path.split('/')[-1]}"
    os.makedirs(root_dir, exist_ok=True)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map=device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Generate text
    max_length = 1
    num_return_sequences = 1

    
    # load dataset dictionary
    with open(args.dataset_name_or_path) as f:
        dataset_dict = json.load(f)
    
    for task in dataset_dict.keys():
        
        
        """
            This is zero-shot dataset but sommtimes when we push dataset object into hub 
            it saves into dataset dictionary with `train` split.
        """
        if args.split_name:
            split = args.split_name
        else:
            split = 'train'
            
        dataset = datasets.load_dataset(dataset_dict[task], split=split)
        print(dataset)
        
        dataset = dataset.select(random.sample(range(len(dataset)), min(args.num_samples, len(dataset))))
        
        # cast audio
        dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    
        
        
        speech_tokenized = pd.read_csv(f"SpeechTokenized/{task}.csv")
    
        predictions = []  
        
        
        
        all_tokens = list(tokenizer.vocab.values())
        
        classes = list(set(dataset['label']))
        options = " ".join([f"{chr(65 + i)}. {class_}" for i, class_ in enumerate(classes)])
        
        labels = [f"{chr(65 + i)}" for i, label in enumerate(classes)]
        # create labels to classses dictionary
        # labels_to_ckass = {label:class_ for label, class_ in zip(labels, classes)}
        class_to_labels = {class_:label for label, class_ in zip(labels, classes)}
        
        suppress_tokens = list(set(all_tokens) - set(tokenizer.convert_tokens_to_ids(labels)))
        
        # iterate over all examples in the test
        for idx in tqdm(range(len(dataset)), desc=f"Evaluation {args.model_name_or_path.split('/')[-1]} for {task}: "):
        
            
            example = dataset[idx]
            
            
            instruction = example['instruction'] + f" Options: {options} Answer:"
            image_tokens = speech_tokenized.loc[idx]['audio_tokens']
            
            prompt_format = PROMPT_DICT['prompt_input_task'].format(task=TASK_DICT['s2t'], instruction=instruction, input=image_tokens)
               
            # import pdb;pdb.set_trace()
            inputs = tokenizer(prompt_format, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}  # Move each tensor to the device
                
            
            prompt_length = inputs['input_ids'].shape[1]
            
            # get the probablities of each class
            # labels = [f"{chr(65 + i)}" for i, label in enumerate(ast.literal_eval(example['options']))]
            # probs = [get_probs(model, inputs, label) for label in [f"{chr(65 + i)}" for i, label in enumerate(ast.literal_eval(example['options']))]]
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,                
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.eos_token_id,  # To avoid warnings for models without pad_token_id
                suppress_tokens=suppress_tokens,
                temperature=0.0,
            )
            answer = tokenizer.decode(outputs[0][prompt_length:])

            predictions.append(answer)
        
        # import pdb;pdb.set_trace()
        y_true = [class_to_labels[label] for label in dataset['label']]
        accuracy = f'{accuracy_score(y_true=y_true, y_pred=predictions)*100:.2f}'
        print(accuracy)
        
        # save the results into json file
        with open(f"{root_dir}/{task}.json", "w") as file:
            json.dump(
                {
                    "task":task,
                    "dataset":args.dataset_name_or_path.split("/")[-1],
                    "model":args.model_name_or_path.split("/")[-1],
                    "num_examples":len(predictions),
                    "accuracy":accuracy,
                },
                file,
                indent = 4
            )
        # save the predictions into csv files
        pd.DataFrame(data={"ground_truth":y_true, "prediction":predictions}).to_csv(f"{root_dir}/{task}.csv", index=False)
        import pdb;pdb.set_trace()

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(prog='eval_mmmu.py',description='zero-shot glam-inference',epilog='Text at the bottom of help')
    parser.add_argument("--model_name_or_path", type=str, default="GLAM24/phi2_baseline_240604_glam_instruct_1m", required=False, help="Provide model name here.",)
    parser.add_argument("--dataset_name_or_path", type=str, default="speech_tasks.json", required=False)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--task_name", type=str, default="mmmu", required=False)
    parser.add_argument("--split_name", type=str, default="test", required=False)
    parser.add_argument("--device", type=str, default=None, required=False)
    parser.add_argument("--output_dir", type=str, default="SpeechResults/",)
    
    args = parser.parse_args()
    main(args=args)