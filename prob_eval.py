import transformers
import argparse
import torch
import datasets
from transformers import AutoProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import json
import pandas as pd
import os


def main(args):
    
    
    device = args.device if torch.cuda.is_available() else "cpu"
    # load datasets
    dataset = datasets.load_dataset(args.dataset_name_or_path, split=args.split_name)
    
    # load the process and tokenizer
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path, device_map=device)
    model.eval()
    
    # get prompt id to get to the model
    prompt_ids = processor.get_prompt_ids(dataset[0]['instruction'], return_tensors="pt").to(device)
    
    # generation kwargs
    # we are going to beam search
    # since we are doing classification we are going to generate only one token (class label). 
    # generation args are hard coded because this is going to be same for all experiments. 
    # "max_new_tokens": num_of_tokens_in_label, "min_new_tokens":num_of_tokens_in_label,
     
    
    gen_kwargs = {
        "num_beams": 1,
        "output_scores":True,
        "output_logits":True,
        "return_dict_in_generate":True,
    }
    
    labels = list(set(dataset['label']))
    
    instruction = dataset[0]['instruction']
    
    empty_string_tokens = processor.tokenizer("", return_tensors="pt") # This is useful to get class tokens. 
    
    
    
    def get_probs(model, input_features, label):
        
        prompt_ids = processor.get_prompt_ids(instruction+f" Answer: {str(label)}", return_tensors="pt").to(device)
        output_ids = processor.tokenizer(str(label), return_tensors="pt")['input_ids'][0][empty_string_tokens['input_ids'].shape[-1]-1:].unsqueeze(0).to(device)
        
        output = model.generate(
            input_features=input_features.to(device), 
            prompt_ids=prompt_ids, 
            max_new_tokens=output_ids.shape[-1],
            min_new_tokens=output_ids.shape[-1],
            **gen_kwargs
        )
        logits = torch.stack(output.logits, dim=1)
        
        # Get the prob for all tokens in the label and accumulate them
        probs = torch.gather(F.softmax(logits, dim=2), 2, output_ids.unsqueeze(2).to(device))
        return probs.sum().detach().cpu()
        
        
    
    predictions = []  
    
    # create a directory for each model to save results
    root_dir = f"{args.output_dir}/{args.model_name_or_path.split('/')[-1]}"
    os.makedirs(root_dir, exist_ok=True)
    
    # iterate over all examples in the test
    for idx in tqdm(range(len(dataset)), desc=f"Evaluation {args.model_name_or_path.split('/')[-1]} for {args.task_name}: "):
        example = dataset[idx]
        input_features = processor(example['audio']['array'], return_tensors="pt").input_features

        
        probs = [get_probs(model, input_features, c) for c in labels]
        
        prediction = labels[np.argmax(probs)]
        predictions.append(prediction)
        # break
    
    
    # import pdb;pdb.set_trace()
    y_true = dataset['label'][:len(predictions)]
    accuracy = accuracy_score(y_true=y_true, y_pred=predictions)
    
    # save the results into json file
    with open(f"{root_dir}/{args.task_name}.json", "w") as file:
        json.dump(
            {
                "task":args.task_name,
                "dataset":args.dataset_name_or_path.split("/")[-1],
                "model":args.model_name_or_path.split("/")[-1],
                "num_examples":len(predictions),
                "accuracy":accuracy,
            },
            file,
            indent = 4
        )
    # save the predictions into csv files
    pd.DataFrame(data={"ground_truth":y_true, "prediction":predictions}).to_csv(f"{root_dir}/{args.task_name}.csv", index=False)

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(prog='whisper_zs.py',description='zero-shot whisper-inference',epilog='Text at the bottom of help')
    parser.add_argument("--model_name_or_path", type=str, default="openai/whisper-large-v2", required=True, help="Provide model name here.",)
    parser.add_argument("--dataset_name_or_path", type=str, default="HaninZ/SpoofDetection_ASVspoof2017_TTS", required=True)
    parser.add_argument("--task_name", type=str, default=None, required=True)
    parser.add_argument("--split_name", type=str, default="test", required=False)
    parser.add_argument("--prompt_file", type=str, default=None, required=False)
    parser.add_argument("--device", type=str, default=None, required=False)
    parser.add_argument("--output_dir", type=str, default="results/",)
    
    args = parser.parse_args()
    main(args=args)