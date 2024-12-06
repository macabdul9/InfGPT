from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datasets
import random


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:{instruction}\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task."
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:{instruction}\n\n### Response:"
    ),
    "prompt_input_task": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Task:{task}\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input_task": (
        "Below is an instruction that describes a task."
        "Write a response that appropriately completes the request.\n\n"
        "### Task:\n{task}\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
# we need to add the values to the tokenizer's vocab to finetune
TASK_DICT = {
    "t2t":"Text2Text",
    "t2i":"Text2Image",
    "i2t":"Image2Text",
    "s2t":"Speech2Text",
    "t2s":"Text2Speech"
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# Load model and tokenizer
model_name = "GLAM24/phi2_baseline_240604_glam_instruct_1m"
# model_name = "GLAM24/GVLAM-Llama-3.1-8B-instruct-1m"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)



# Generate text
max_length = 512
num_return_sequences = 1

dataset = datasets.load_dataset("GLAM24/GVLAM-Instruct-Eval")




while True:
    
    idx = random.randint(0, len(dataset['test']))
    
    example = dataset['test'][idx]
    
    prompt = PROMPT_DICT['prompt_input_task'].format(task=TASK_DICT[example['id'].split("_")[0]], instruction=example['instruction'], input=example['input']) if example.get("input", "") != "" else PROMPT_DICT['prompt_no_input_task'].format(task=TASK_DICT[example['id'].split("_")[0]], instruction=example['instruction'])
    
    # import pdb;pdb.set_trace()
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move each tensor to the device

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,  # To avoid warnings for models without pad_token_id
        temperature=0.7,
        top_k=100,
        top_p=0.97,
        do_sample=True,  # For more diverse outputs
    )

    # Decode and print the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(generated_text)
    
    import pdb;pdb.set_trace()