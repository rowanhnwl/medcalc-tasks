import pandas as pd
import json
from tqdm import tqdm
import os
import argparse
import re

from utils import *
from medcalc import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--save_dir", type=str, required=False, default="results")
    parser.add_argument("--dataset_path", type=str, required=False, default="dataset/test.csv")

    args = parser.parse_args()

    test_path = args.dataset_path
    model_path = args.model
    save_dir = args.save_dir

    medcalc_df = pd.read_csv(test_path)

    system_prompt = PromptPrefix.ZERO_SHOT

    model, tokenizer = load_model_and_tokenizer(model_path)

    categories_dict = {}

    for datapoint in medcalc_df.iloc:
        datum = MedCalcDatum(datapoint)
        
        try:
            categories_dict[datum.category].append(datum)
        except:
            categories_dict[datum.category] = [datum]

    results_dict = {category: 0 for category in categories_dict}

    for category in categories_dict:
        for datum in tqdm(categories_dict[category], desc=category):
            prompt = datum.create_prompt()

            correctness = 0

            answer = generate_answer(system_prompt, prompt, model, tokenizer)

            answer = re.search(r"\{.*?\}", answer, flags=re.DOTALL)
            try:
                answer = json.loads(answer.group(0))
                
                correctness = check_correctness(
                    str(answer["answer"]),
                    datum.ground_truth_answer,
                    datum.calculator_id,
                    datum.upper_limit,
                    datum.lower_limit
                )
            except:
               print("Could not parse the answer")

            results_dict[category] += correctness

        results_dict[category] /= len(categories_dict[category])

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "zero_shot_results.json")
    with open(save_path, "w") as f: json.dump(results_dict, f, indent=3)

if __name__ == "__main__":
    main()