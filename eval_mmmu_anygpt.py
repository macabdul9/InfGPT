import sys
sys.path.append("./")
sys.path.append("./anygpt/src")
import os
import torch
import torch.nn.functional as F
import torchaudio
from einops import rearrange
import argparse
import logging
import json
import re
import numpy as np
import traceback
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, EncodecModel, AutoProcessor
from seed2.seed_llama_tokenizer import ImageTokenizer
from PIL import Image
from datetime import datetime
from speechtokenizer import SpeechTokenizer
from m_utils.anything2token import *
from m_utils.read_modality import encode_music_by_path
from m_utils.conversation import get_conv_template
from voice_clone import load_soundstorm, semantic2acoustic
from infer.pre_post_process import extract_content_between_final_tags
from m_utils.prompter import *
import datasets
from tqdm import tqdm
import ast
import json



logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnyGPTInference:
    def __init__(
        self,
        model_name_or_path: str,
        image_tokenizer_path: str,
        output_dir: str,
    ):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

                # model
        print("loading llm")
        self.model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            )
        # self.model.half()  
        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)
        #tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path)
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left" 
        self.output_dir = output_dir

        print('Loading image tokenizer')
        self.image_tokenizer = ImageTokenizer(model_path=image_tokenizer_path, load_diffusion=True,
                                                  diffusion_model_path="stabilityai/stable-diffusion-2-1-unclip", device=self.device, image_size=224)

        self.prompter = Prompter()

    def encode_image(
        self,
        image_path=None,
        image_pil=None,
        image_torch=None
    ):
        assert (image_path is None) + (image_pil is None) + (image_torch is None) == 2

        # need_norm_to_1 = False
        if image_path is not None:
            image_pil = Image.open(image_path).convert('RGB')

        if image_pil is not None:
            image_torch = self.image_tokenizer.processor(image_pil)

            image_torch = image_torch.to(self.device)
        return self.image_tokenizer.encode(image_torch)
    
    
    def preprocess(
        self,
        task, instruction, 
        image_files=None,
        speech_files=None,
        music_files=None
    ):
        image_list=[]

        for image in image_files:
            tokens = self.encode_image(image_pil= image)[0]
            processed_inputs = modality_tokens_to_string(tokens=tokens, modality="image")
            # print("image: ", processed_inputs)
            image_list.append(processed_inputs)
        
        # 使用sft_prompt
        prompt_seq = self.prompter.generate_insturction_prompt(task,instruction,image_list,[],[]).strip()
        return prompt_seq

    def get_probs(self, inputs, label):
        
        output_ids = self.tokenizer(str(label), return_tensors="pt", add_special_tokens=False, padding=True).input_ids.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids= inputs,
                max_new_tokens=output_ids.shape[-1], 
                min_new_tokens=output_ids.shape[-1], 
                num_beams=1,
                output_scores=True,
                output_logits=True,  # Allows collection of token scores
                return_dict_in_generate=True  # Returns GenerateOutput object
            )
        logits = torch.stack(output_ids.logits, dim=1)
        
        # Calculate the probabilities for all tokens in the label
        probs = torch.gather(F.softmax(logits, dim=2), 2, output_ids.unsqueeze(2).to(device))
        return probs.sum().detach().cpu().item()
    
    def response(self, task, instruction, image_files, options):
        preprocessed_prompts = (self.preprocess(task, instruction, image_files, [], []))
        input_ids = self.tokenizer(preprocessed_prompts, return_tensors="pt", padding=True).input_ids
        input_ids = input_ids.to(self.device)

        probs = [self.get_probs(input_ids, label) for label in [f"{chr(65 + i)}" for i, label in enumerate(ast.literal_eval(options))]]

        return np.argmax(probs)

if __name__ == '__main__':
    ds = datasets.load_dataset('MMMU/MMMU', 'Agriculture', split= 'validation')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="output_models/visual_inter_speech_golden_fs/checkpoint-30000")
    parser.add_argument("--image-tokenizer-path", type=str, default="models/seed-tokenizer-2/seed_quantizer.pt")
    parser.add_argument("--output-dir", type=str, default="infer_output/test")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    infer = AnyGPTInference(
        args.model_name_or_path,
        args.image_tokenizer_path,
        args.output_dir
    )
    predictions = []
    for idx in tqdm(range(len(ds)), desc=f"Evaluation {args.model_name_or_path.split('/')[-1]} for Agriculture: "):
        row = ds[idx]
        ## ignoring multilple images
        if row['image_2'] != None:
            continue
        question = row['question']
        options = row['options']
        image_1 = row['image_1']
        answer = row['answer']

        labels = [f"{chr(65 + i)}" for i, label in enumerate(ast.literal_eval(options))]
        options_str = " ".join([f"{chr(65 + i)}. {label}" for i, label in enumerate(ast.literal_eval(options))])
        instruction = question + f' Options: {options_str}. Answer:'
        response = infer.response('Image QA', instruction, [image_1], options)

        predictions.append(labels[response])
    
    y_true = ds['answer']#[:len(predictions)]
    accuracy = accuracy_score(y_true=y_true, y_pred=predictions)
    print(accuracy)

    with open(f"{args.output_dir}/{'Agriculture'}.json", "w") as file:
        json.dump(
            {
                "task":config,
                "dataset":args.dataset_name_or_path.split("/")[-1],
                "model":args.model_name_or_path.split("/")[-1],
                "num_examples":len(predictions),
                "accuracy":accuracy,
            },
            file,
            indent = 4
        )
        # save the predictions into csv files
        pd.DataFrame(data={"ground_truth":y_true, "prediction":predictions}).to_csv(f"{args.output_dir}/{'Agriculture'}.csv", index=False)