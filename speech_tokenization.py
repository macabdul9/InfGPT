import datasets
import torch
from speechtokenizer import SpeechTokenizer
import pandas as pd
from tqdm import tqdm
import random


def main(args):
    
    device = args.device
    
    speech_tokenizer = SpeechTokenizer.load_from_checkpoint(
        f'{args.speech_tokenizer_name_or_path}/config.json', 
        f'{args.speech_tokenizer_name_or_path}/SpeechTokenizer.pt'
    ).to(args.device)
    
    # for normal datasets
    # dataset = datasets.load_dataset(args.dataset_name_or_path, split=args.split)
    # dataset = dataset.select(random.sample(range(len(dataset)), min(args.num_samples, len(dataset))))
    
    # LibriTTS datasets
    dataset = datasets.load_dataset(args.dataset_name_or_path, "clean")
    dataset = datasets.concatenate_datasets([dataset['train.clean.100'], dataset['train.clean.360']])
    # cast audio
    dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    

    
    inputs = []
    outputs = []
    
    for example in tqdm(dataset):
        
        speech = torch.tensor(example['audio']['array'], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # ensure that audio is sampled at 16K 
        
        # 1. Get discrete speech tokens. 1.1 Take semantic tokens from all speech tokens. 
        with torch.no_grad():
            speech_tokenized = speech_tokenizer.encode(speech.to(device)).cpu() # codes: (n_q, B, T)
            
        semantic_tokens = speech_tokenized[:1, :, :].flatten().tolist() # Contain content info, can be considered as semantic tokens
    
        # 1.2. Convert these discrete tokens into audio tokens as specified in GLAM tokenizer. 
        ## Adding BOS token before audio tokens as recommended by Hao!
        # input_ = "".join(["<aud>"] + ['<aud_{:05d}>'.format(token) for token in semantic_tokens] + ["</aud>"])
        
        audio_tokens = "".join(['<aud_{:05d}>'.format(token) for token in semantic_tokens])
        
        
        # import pdb;pdb.set_trace()
        # if args.text_input:
        #     input_ = f'{input_} Text: {example[args.text_input_name]}'
        
        
        inputs.append(example['text_original'])
        outputs.append(audio_tokens)
        
        # inputs.append(f'{example["instruction"]} Speech: {input_}')
        # outputs.append(example['label'])
        
        # import pdb;pdb.set_trace()
        
    new_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data={"input":inputs, "output":outputs}))
    new_dataset.save_to_disk(f'{args.output_dir}/{args.dataset_name_or_path.split("/")[-1]}_{args.num_samples}')
    if args.push_to_hub:
        new_dataset.push_to_hub(f'{args.dataset_name_or_path.split("/")[-1]}_{args.num_samples}', private=True)


if __name__=="__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name_or_path", type=str, default="DynamicSuperbPrivate/SpeechTextMatching_Tedlium2Train")
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--text_input", type=bool, default=None)
    parser.add_argument("--text_input_name", type=str, default="transcription")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--speech_tokenizer_name_or_path", type=str, default="GLAM24/phi2_baseline_movq_speechtokenizer_init_data_240403", help="Provide tokenizer path (loca) or name (huggingface).",)
    parser.add_argument("--device", type=str, default=0)
    parser.add_argument("--output_dir", type=str, default="./data/tokenized", help="")
    parser.add_argument("--push_to_hub", default=None, help="Whether to push final tokenizer on huggingface or not. Pass the hf name if True. org/user / name.")
    
    args = parser.parse_args()
    main(args)
    
    # def format_data(example):
    #     example['output'] = "<aud>"+example['output']+"</aud>"
    #     example['input'] = f"{random.choice(prompts)} Text: {example['input']}"
    #     return example
