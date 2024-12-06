import datasets
import torch
from speechtokenizer import SpeechTokenizer
import pandas as pd
from tqdm import tqdm
import random
import json


def main(args):
    
    device = args.device
    
    speech_tokenizer = SpeechTokenizer.load_from_checkpoint(
        f'{args.speech_tokenizer_name_or_path}/config.json', 
        f'{args.speech_tokenizer_name_or_path}/SpeechTokenizer.pt'
    ).to(args.device)
    
    
    
    # load dataset dictionary
    with open(args.dataset_name_or_path) as f:
        dataset_dict = json.load(f)
    
    for task in dataset_dict.keys():
        if args.split:
            split = args.split
        else:
            split = 'train'
            
        dataset = datasets.load_dataset(dataset_dict[task], split=split)
        print(dataset)
        # continue
        
        dataset = dataset.select(random.sample(range(len(dataset)), min(args.num_samples, len(dataset))))
        
        # cast audio
        dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    
        
        audio_tokens_all = []
        
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
            
            
            # inputs.append(example['text_original'])
            audio_tokens_all.append(audio_tokens)
            
            # inputs.append(f'{example["instruction"]} Speech: {input_}')
            # outputs.append(example['label'])
            
            # import pdb;pdb.set_trace()
            
        ids = [f"{task}_{i}" for i in range(len(audio_tokens_all))]
        df = pd.DataFrame(data={"id":ids, "audio_tokens":audio_tokens_all})
        df.to_csv(f"SpeechTokenized/{task}.csv", index=False)
        


if __name__=="__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name_or_path", type=str, default="speech_tasks.json")
    parser.add_argument("--speech_tokenizer_name_or_path", type=str, default="speechtokenizer_hubert_avg/", help="Provide tokenizer path (loca) or name (huggingface).",)
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default=0)
    parser.add_argument("--output_dir", type=str, default="SpeechTokenized/", help="")
    
    args = parser.parse_args()
    main(args)
