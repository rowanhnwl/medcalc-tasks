import argparse

from transformers import TrainingArguments, Trainer

from utils import *
from medcalc import *

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--lora", action='store_true')

    parser.add_argument("--train_dataset", type=str, required=False, default="dataset/train_split.csv")
    parser.add_argument("--valid_dataset", type=str, required=False, default="dataset/valid_split.csv")
    parser.add_argument("--lr", type=float, required=False, default=1e-4)
    parser.add_argument("--epochs", type=float, required=False, default=3)

    args = parser.parse_args()

    model_path = args.model
    train_dataset_path = args.train_dataset
    valid_dataset_path = args.valid_dataset
    lora = args.lora

    lr = args.lr
    epochs = args.epochs
    save_dir = args.save_path

    if lora:
        model, tokenizer = load_lora_model_and_tokenizer(hf_path=model_path)
    else:
        model, tokenizer = load_model_and_tokenizer(hf_path=model_path)

    train_dataset = MedCalcFineTuneDataset(dataset_path=train_dataset_path, tokenizer=tokenizer)
    valid_dataset = MedCalcFineTuneDataset(dataset_path=valid_dataset_path, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        learning_rate=lr,
        bf16=True,
        logging_dir="./logs",
        report_to="none",
        label_names=["labels"],
        eval_strategy="steps",
        eval_steps=500,
        per_device_eval_batch_size=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )

    data_collator = MaskingCollator(
        tokenizer=tokenizer
    )

    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator
    )

    trainer.train()

if __name__ == "__main__":
    main()