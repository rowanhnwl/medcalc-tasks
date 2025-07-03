import json
import pandas as pd
from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from transformers import AutoTokenizer

class PromptPrefix:
    ZERO_SHOT = 'You are a helpful assistant for calculating a score for a given patient note. Please output answer only without any other text. Your output should only contain a JSON dict formatted as {"answer": "<ANSWER>"}.'
    
    FEW_SHOT = "You are a helpful medical tool being used for clinical calculation tasks. Analyze the following examples to make the specified calculation at the end.\n \
        Please format your response as a JSON structured as \"{\"answer\": \"<ANSWER>\"}\" where <ANSWER> is a single value.\n\n"
    
    TRAINING_PROMPT = 'You are a helpful assistant for calculating a score for a given patient note. Please output a brief explanation of the calculation process along with the final answer.'

class MedCalcDatum:

    LABEL_KEYS = [
        "Ground Truth Answer",
        "Lower Limit",
        "Upper Limit",
        "Ground Truth Explanation"
    ]

    def __init__(self, datapoint):
        self.information = {k: v for (k, v) in datapoint.items()}

        self.id = self.information["Row Number"]
        self.patient_note = self.information["Patient Note"]
        self.question = self.information["Question"]
        self.category = self.information["Category"]
        self.calculator_id = self.information["Calculator ID"]
        self.ground_truth_answer = self.information["Ground Truth Answer"]
        self.lower_limit = self.information["Lower Limit"]
        self.upper_limit = self.information["Upper Limit"]
        self.ground_truth_explanation = self.information["Ground Truth Explanation"]

        self.format_relevant_entities()

    def format_relevant_entities(self):
        self.information["Relevant Entities"] = json.loads(self.information["Relevant Entities"].replace("\'", "\"").replace("True", "true").replace("False", "false"))

    def create_prompt(self, infer=False):
        prompt = f'Here is the patient note: \n{self.patient_note}\n\nHere is the task:\n{self.question}\n\nnPlease directly output the JSON dict formatted as {{"answer": "<ANSWER>"}}:'

        return prompt
    
    def create_trainable_prompt(self, infer=False):
        prompt = self.information["Patient Note"] + "\n\n" + self.information["Question"]
        
        if not infer:
            prompt += "\n\n" + "Ground Truth Explanation: " + self.ground_truth_explanation

        return prompt
    
class MedCalcFewShotPrompt:
    def __init__(self, test_datum: MedCalcDatum, example_data: list[MedCalcDatum]):
        self.test_datum = test_datum
        self.example_data = example_data

        self.prompt = self.create_prompt()

    def create_prompt(self):
        prompt = f""

        for i, example in enumerate(self.example_data):
            prompt += f"EXAMPLE {i}: \n\n"
            prompt += example.create_prompt()

        prompt += "EXAMPLE TO BE ANSWERED:\n\n"
        prompt += self.test_datum.create_prompt(infer=True)

        return prompt
    
class MedCalcFineTuneDataset(Dataset):
    def __init__(self, dataset_path, system_prompt: str, tokenizer: AutoTokenizer, max_length=3072):
        self.medcalc_df = pd.read_csv(dataset_path)
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = self.load_and_tokenize()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def load_and_tokenize(self):
        tokenized_data = []

        for row in tqdm(self.medcalc_df.iloc, desc="tokenizing the dataset"):
            datum = MedCalcDatum(row)
            
            text = datum.create_trainable_prompt()
            text_no_explanation = datum.create_trainable_prompt(infer=True)

            tokenized_text = self.tokenize(text)
            tokenized_no_exp = self.tokenize(text_no_explanation)

            len_no_exp = len(tokenized_no_exp["input_ids"])
            
            tokenized_text["begin_loss_index"] = len_no_exp - 1

            tokenized_data.append(tokenized_text)

        return tokenized_data
    
    def tokenize(self, text):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )

        tokenized_text = self.tokenizer(text, max_length=self.max_length, truncation=True)

        return tokenized_text

class MaskingCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [torch.tensor(example["input_ids"]) for example in batch]
        attention_mask = [torch.ones_like(ids) for ids in input_ids]

        labels = []
        for example in batch:
            label_ids = example["input_ids"].copy()
            begin_loss_idx = example["begin_loss_index"]

            for i in range(begin_loss_idx + 1):
                label_ids[i] = -100
            labels.append(torch.tensor(label_ids))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }