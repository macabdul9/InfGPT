nohup python eval_mmmu_gen.py --model_name_or_path GLAM24/phi2_baseline_240604_glam_instruct_1m --device cuda:0 &> logs/infgpt_mmmu_gen.log &
nohup python eval_mmmu_gen.py --model_name_or_path GLAM24/phi2_baseline_movq_speechtokenizer_init_data_simple_briging_prompt_240604 --device cuda:1 &> logs/phi2_baseline_pretrained_mmmu_gen.log &
nohup python eval_mmmu_gen.py --model_name_or_path microsoft/phi-2 --device cuda:0 &> logs/phi2_mmmu_gen.log &