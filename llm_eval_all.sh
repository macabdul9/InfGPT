nohup python llm_eval.py \
    --root_dir="MMMUGenResults" \
    --instruction_column="question" \
    --options_column="options" \
    --answer_column="answer" \
    --generated_answer_column="generated_answer" &> logs/MMMUGenResults_eval.log &

nohup python llm_eval.py \
    --root_dir="SpeechGenResults" \
    --instruction_column="question" \
    --options_column="options" \
    --answer_column="answer" \
    --generated_answer_column="generated_answer" &> logs/SpeechGenResults_eval.log &