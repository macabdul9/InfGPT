import transformers
import argparse
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
import json
import os
import random
from sklearn.metrics import accuracy_score
import sys
# sys.path.append("./")
# sys.path.append("./anygpt/src")

# Prompt Templates
PROMPT_DICT = {
    "prompt_input_task": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Task:{task}\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
}

# Task Mapping
TASK_DICT = {
    "t2t": "Text2Text",
    "t2i": "Text2Image",
    "i2t": "Image2Text",
    "s2t": "Speech2Text",
    "t2s": "Text2Speech"
}

def main(args):
    device = args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create output directory
    root_dir = f"{args.output_dir}/{args.model_name_or_path.split('/')[-1]}"
    os.makedirs(root_dir, exist_ok=True)
    
    # Generate text
    max_length = 512
    num_return_sequences = 1
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map=device,torch_dtype=torch.float16).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Load dataset dictionary
    # with open(args.dataset_name_or_path) as f:
    #     dataset_dict = json.load(f)
    dataset_dict = {
        "MultiSpeakerDetection": "macabdul9/MultiSpeakerDetection_VCTK",
        "IntentClassification": "macabdul9/IntentClassification_FluentSpeechCommands-Action",
        "AccentClassification": "macabdul9/AccentClassification_AccentdbExtended",
        "SpeechDetection": "macabdul9/SpeechDetection_LibriSpeech-TestClean",
        "BirdSoundDetection": "macabdul9/BirdSoundDetection_Warblrb10k",
        "EmotionRecognition": "macabdul9/EmotionRecognition_MultimodalEmotionlinesDataset",
        "SpeechTextMatching": "macabdul9/SpeechTextMatching_LJSpeech",
        "LanguageIdentification": "macabdul9/LanguageIdentification_VoxForge"
        "SarcasmDetection": "macabdul9/SarcasmDetection_Mustard",
        "NoiseDetection": "macabdul9/NoiseDetection_LJSpeech_MUSAN-Gaussian",
        "ChordClassification": "macabdul9/ChordClassification_AcousticGuitarAndPiano"
    }

    for task in dataset_dict.keys():
        # Determine split
        split = args.split_name if args.split_name else "train"
        
        # Load dataset
        dataset = datasets.load_dataset(dataset_dict[task], split=split)
        dataset = dataset.select(random.sample(range(len(dataset)), min(args.num_samples, len(dataset))))
        
        # Cast audio to ensure uniform sampling rate
        dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
        
        # Load tokenized data for the task
        speech_tokenized = pd.read_csv(f"SpeechTokenized/{task}.csv")

        predictions = []
        options_all = []
        classes = list(set(dataset["label"]))
        options = " ".join([f"{chr(65 + i)}. {class_}" for i, class_ in enumerate(classes)])
        
        # Label to class mapping
        labels = [f"{chr(65 + i)}" for i, label in enumerate(classes)]
        class_to_labels = {class_: label for label, class_ in zip(labels, classes)}
        
        # Iterate over dataset examples
        for idx in tqdm(range(len(dataset)), desc=f"Evaluation {args.model_name_or_path.split('/')[-1]} for {task}: "):
            
            example = dataset[idx]
            
            # Generate instruction and input tokens
            instruction = example["instruction"] + f" Answer:"
            
            audio_tokens = speech_tokenized.loc[idx]["audio_tokens"]
            
            prompt_format = PROMPT_DICT["prompt_input_task"].format(task=TASK_DICT["s2t"],instruction=instruction,input=audio_tokens)
            
            # import pdb;pdb.set_trace()
            # Tokenize inputs
            inputs = tokenizer(prompt_format, return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            
            prompt_length = inputs['input_ids'].shape[1]

            # # Generate output
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.0,
            )
            
            # outputs = model.generate(
            #     **inputs,
            #     max_new_tokens=512,
            #     num_return_sequences=num_return_sequences,
            #     pad_token_id=tokenizer.eos_token_id,  # To avoid warnings for models without pad_token_id
            #     temperature=0.7,
            #     top_k=100,
            #     top_p=0.97,
            #     do_sample=True,  # For more diverse outputs
            # )
            
            answer = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
            # import pdb;pdb.set_trace()
            
            # import pdb;pdb.set_trace()
            
            predictions.append(answer)
            options_all.append(options)
            
        # Save predictions
        # save the predictions into csv files
        pd.DataFrame(data={"question":dataset['instruction'], "options":options_all, "answer":dataset['label'], "generated_answer":predictions}).to_csv(f"{root_dir}/{task}.csv", index=False)
        # import pdb;pdb.set_trace()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog="eval_speech_tasks.py",
        description="Evaluate multiple speech tasks with a zero-shot model",
        epilog="Zero-shot speech task evaluation script."
    )
    parser.add_argument("--model_name_or_path", type=str, default="fnlp/AnyGPT-base", help="Model name or path.")
    parser.add_argument("--dataset_name_or_path", type=str, default="speech_tasks.json", help="Path to dataset JSON file.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to evaluate per task.")
    parser.add_argument("--split_name", type=str, default="test", help="Dataset split to evaluate.")
    parser.add_argument("--device", type=str, default=None, help="Device to use for evaluation.")
    parser.add_argument("--output_dir", type=str, default="SpeechGenResults/", help="Directory to save results.")
    
    args = parser.parse_args()
    main(args)