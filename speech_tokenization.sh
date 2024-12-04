#!/bin/bash

export HF_HOME=$HOME/.cache
export HF_DATASETS_OFFLINE=0
export HF_DATASETS_CACHE=$HOME/.cache

nohup python speech_tokenization.py \
    --dataset_name_or_path "DynamicSuperb/Text2Speech_LibriTTS-TestOther" \
    --split "test" \
    --num_samples 1000000 \
    --speech_tokenizer_name_or_path "SpeechTokenizer/speechtokenizer_hubert_avg" \
    --device "cuda:7" \
    --output_dir "./data/multispeaker-tts" \
    --push_to_hub False \
    --text_input True &> logs/Text2Speech_LibriTTS-TestOther.log &

nohup python speech_tokenization.py \
    --dataset_name_or_path "DynamicSuperb/Text2Speech_LibriTTS-TestClean" \
    --split "test" \
    --num_samples 1000000 \
    --speech_tokenizer_name_or_path "SpeechTokenizer/speechtokenizer_hubert_avg" \
    --device "cuda:7" \
    --output_dir "./data/multispeaker-tts" \
    --push_to_hub False \
    --text_input True &> logs/Text2Speech_LibriTTS-TestClean.log &