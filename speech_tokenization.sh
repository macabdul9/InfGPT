#!/bin/bash

# export HF_HOME=$HOME/.cache
# export HF_DATASETS_OFFLINE=0
# export HF_DATASETS_CACHE=$HOME/.cache

# Download the SpeechTokenizer model from the Hugging Face Hub
mkdir SpeechTokenizer
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('fnlp/SpeechTokenizer', repo_type='model', local_dir='./SpeechTokenizer')
"

mkdir SpeechTokenized
python speech_tokenization.py --dataset_name_or_path "speech_tasks.json" --speech_tokenizer_name_or_path SpeechTokenizer/speechtokenizer_hubert_avg --output_dir SpeechTokenized