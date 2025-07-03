import pandas as pd
import json
from tqdm import tqdm
import random
import argparse
import os
import re

from utils import *
from medcalc import *

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--shots", type=int, required=True)

    parser.add_argument("--save_dir", type=str, required=False, default="results")
    parser.add_argument("--dataset_path", type=str, required=False, default="dataset/test.csv")

    args = parser.parse_args()

    test_path = args.dataset_path
    model_path = args.model
    n_shots = args.shots
    save_dir = args.save_dir

    prompt_prefix = PromptPrefix.FEW_SHOT

    medcalc_df = pd.read_csv(test_path)

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
            
            shots = []
            while len(shots) < n_shots:
                shot = random.sample(categories_dict[category], k=1)[0]
                # ensure that the other shots are different but of the same calculator type
                if shot.id != datum.id and shot.calculator_id == datum.calculator_id:
                    shots.append(shot)

            few_shot_obj = MedCalcFewShotPrompt(test_datum=datum, example_data=shots)
            prompt = prompt_prefix + few_shot_obj.prompt

            answer = generate_answer(prompt, model, tokenizer)
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
                continue

            results_dict[category] += correctness

        results_dict[category] /= len(categories_dict[category])

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{n_shots}_shots_results.json")
    with open(save_path, "w") as f: json.dump(results_dict, f, indent=3)

if __name__ == "__main__":
    main()