# MedCalc Benchmark Tasks

This repository explores the MedCalc benchmark with the Qwen3-1.7B and Qwen3-4B models for:

- Zero-shot and few-shot inference
- Paremeter-efficient fine-tuning
- Full fine-tuning

## Environment

All tasks were run using an A100-40GB GPU.

```
conda create -n medcalc python
conda activate medcalc
pip install -r requirements.txt
```

## Dataset

The dataset is provided in the repository at `dataset/` and includes the training set (`train.csv`), testing set (`test.csv`), and the training and validation splits (`train_split.csv` and `test_split.csv`).

## Zero-shot inference

For zero-shot inference, run the following command:

```
python zeroshot.py --model <model_path> --save_dir <path_to_save_directory> --dataset_path <path_to_dataset>
```

The `--save_dir` and `--dataset_path` arguments are optional, with the default `save_dir` being `results/`.

## Few-shot inference

For few-shot inference, run the following command:

```
python fewshot.py --model <model_path> --shots <number_of_shots> --save_dir <path_to_save_directory> --dataset_path <path_to_dataset>
```

## Fine-tuning

To fine-tune a Qwen3 model on the training dataset, run the following command:

```
python finetune.py --model <model_path> --save_path <path_to_save_model> --lora --train_dataset <path_to_training_dataset> --valid_dataset <path_to_validation_dataset> --lr <learning_rate> --epochs <number_of_epochs>
```

The `--lora` flag specifies whether or not LoRA will be used. The only required arguments are `--model` and `--save_path`. The default `lr` is `0.0001` and the default `epochs` is `3`. To conserve memory, the batch size is fixed at 1.