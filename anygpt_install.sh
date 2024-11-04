#clone the repo
git clone https://github.com/OpenMOSS/AnyGPT.git
cd AnyGPT
conda create --name AnyGPT python=3.9
conda activate AnyGPT
pip install -r requirements.txt

#fix huggingface-hub
pip install --upgrade huggingface-hub==0.23

#install git lfs
sudo apt-get install git-lfs
git lfs install

#download tokenizers and such
mkdir models/
cd models
git clone https://huggingface.co/fnlp/AnyGPT-speech-modules
git clone https://huggingface.co/AILab-CVC/seed-tokenizer-2

#quick test
cd ..
python anygpt/src/infer/cli_infer_base_model.py \
--model-name-or-path fnlp/AnyGPT-base \
--image-tokenizer-path models/seed-tokenizer-2/seed_quantizer.pt \
--speech-tokenizer-path models/AnyGPT-speech-modules/speechtokenizer/ckpt.dev \
--speech-tokenizer-config models/AnyGPT-speech-modules/speechtokenizer/config.json \
--soundstorm-path models/AnyGPT-speech-modules/soundstorm/speechtokenizer_soundstorm_mls.pt \
--output-dir "infer_output/base" 

## this should download all related configs, extra tokens and the llm itself and store them in 
## ~/.cache/huggingface/
## if all works, you can provide it with prompts :) but in a specific format highlighted in the
## anygpt repo OpenMOSS/AnyGPT

