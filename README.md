### Our model - https://huggingface.co/macabdul9/GVLAM-Phi2


```
IGNORE_INDEX = -100

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
```


Steps
1. Tokenize the data for each config
2. Run the eval for that config