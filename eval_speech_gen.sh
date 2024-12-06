# evaluate all speech models
nohup python eval_speech_gen.py --model_name_or_path GLAM24/phi2_baseline_240604_glam_instruct_1m --device cuda:0 &> logs/infgpt_speech_gen.log &
nohup python eval_speech_gen.py --model_name_or_path GLAM24/phi2_baseline_movq_speechtokenizer_init_data_simple_briging_prompt_240604 --device cuda:1 &> logs/phi2_baseline_pretrained_speech_gen.log &
nohup python eval_speech_gen.py --model_name_or_path microsoft/phi-2 --device cuda:1 &> logs/phi2_speech_gen.log &