import os
import json
from typing import Optional
from dataclasses import dataclass, field
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoTokenizer
from torchvision import transforms
import datasets

from external.MoVQ.movq_inference import get_movqgan_model

from PIL import Image

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

# Define the image transformation
img_transform_fn = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def transform_img(example):
    # Convert PIL image to RGB and then apply the transformations
    example['image_1'] = img_transform_fn(example['image_1'].convert('RGB'))
    return example

@torch.no_grad()
def tokenize_image(images, tokenizer, device):
    batch_images = images.to(device)
    batch_embds, _, (_, _, batch_codes) = tokenizer.encode(batch_images)
    batch_codes = batch_codes.view(len(images), -1).cpu().numpy()
    
    image_ids_list = []
    
    for i in range(batch_codes.shape[0]):
        image_ids = batch_codes[i]
        image_ids = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids]) + EOI_TOKEN
        image_ids_list.append(image_ids)
    return image_ids_list

def main():
    device = torch.device("cuda:0")
    
    # Load tokenizer
    image_tokenizer = get_movqgan_model('270M', pretrained=True, device=device)

    # Load data
    configs = ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']
    for config in configs:
        
        dataset = dataset = datasets.load_dataset("MMMU/MMMU", config, split="validation")#.select(range(100))
        
        dataset = dataset.map(transform_img, num_proc=16)  # Parallelize map if possible
        
        dataset.set_format("pt", columns=["image_1"], output_all_columns=True)
        
        if os.path.exists(f"{config}.csv"):
            df.to_csv(f"MMMUTokenized/{config}.csv", index=False)
        else:
            df = pd.DataFrame()
        
        with torch.no_grad():  
            for idx in tqdm(range(len(df), len(dataset), 1), desc="Processing Batches"):
                
                data = {}
                
                data['id'] = f'i2t_{idx}'
                
                # import pdb;pdb.set_trace()
                example = dataset[idx]
                images = example['image_1'].unsqueeze(0)
                
                # for entry in example['texts']:
                #     if entry['from'] == 'human':
                #         data['prompt'] = entry['value'].split("<image>")[-1].strip()
                #     elif entry['from'] == 'gpt':
                #         data['response'] = entry['value']
                    
                image_ids_list = tokenize_image(images=example['image_1'].unsqueeze(0), tokenizer=image_tokenizer, device=device)
                
            
                data.update({"image":image_ids_list[0]})
                
                df = df._append(data, ignore_index=True)
                if idx%5000==0:
                    df.to_csv(f"MMMUTokenized/{config}.csv", index=False)
                    
                    
        
        # df.to_csv("image2text.csv", index=False)
        df.to_csv(f"MMMUTokenized/{config}.csv", index=False)

if __name__ == '__main__':
    main()