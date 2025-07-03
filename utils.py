from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch

import re

from datetime import datetime

def load_model_and_tokenizer(hf_path):

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=hf_path,
        device_map="auto",
        torch_dtype="auto"
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=hf_path
    )

    print(model.device)

    return model, tokenizer

def load_lora_model_and_tokenizer(hf_path):
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        hf_path,
        device_map="auto",
        trust_remote_code=True
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, tokenizer

def generate_answer(system, prompt, model, tokenizer, max_new_tokens=32):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )

    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except:
        index = 0
        
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return content

# from: https://github.com/ncbi-nlp/MedCalc-Bench/blob/main/evaluation/evaluate.py
def check_correctness(answer: str, ground_truth, calid, upper_limit, lower_limit):

    calid = int(calid)

    if calid in [13, 68]:
        # Output Type: date

        if datetime.strptime(answer, "%m/%d/%Y").strftime("%-m/%-d/%Y") == datetime.strptime(ground_truth, "%m/%d/%Y").strftime("%-m/%-d/%Y"):
            correctness = 1
        else:
            correctness = 0
    elif calid in [69]:
        # Output Type: integer (A, B)
        match = re.search(r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", ground_truth)
        ground_truth = f"({match.group(1)}, {match.group(3)})"
        match = re.search(r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", answer)
        if match:
            weeks = match.group(1)
            days = match.group(3)
            answer = f"({weeks}, {days})"
            if eval(answer) == eval(ground_truth):
                correctness = 1
            else:
                correctness = 0
        else:
            correctness = 0
    elif calid in [4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48, 51, 69]:
        # Output Type: integer A
        answer = round(eval(answer))
        if answer == eval(ground_truth):
            correctness = 1
        else:
            correctness = 0
    elif calid in [2,  3,  5,  6,  7,  8,  9, 10, 11, 19, 22, 23, 24, 26, 30, 31, 38, 39, 40, 44, 46, 49, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]:
        # Output Type: decimal
        answer = eval(answer)
        if answer >= eval(lower_limit) and answer <= eval(upper_limit):
            correctness = 1
        else:
            correctness = 0
    else:
        raise ValueError(f"Unknown calculator ID: {calid}")
    return correctness