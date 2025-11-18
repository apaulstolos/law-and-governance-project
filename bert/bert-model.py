import pandas as pd
from datasets import Dataset, ClassLabel
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
import evaluate
import optuna


df = pd.read_csv("/home/msd5170/paul/law/northmet-feis-adequacy-exhibit-a.csv")

df.columns = [
    "Name",
    "Comment",
    "Issue",
    "Substantive_NonSubstantive",
    "Old_New",
    "Response_ID",
    "RGU_Consideration"
]

df = df[df["Comment"].notna()]
df = df[df["Comment"].astype(str).str.strip() != ""]

df = df[df["Substantive_NonSubstantive"].notna()]
df = df[df["Substantive_NonSubstantive"].astype(str).str.strip() != ""]

label_mapping = {
    "S": 1,
    "NS": 0,
    "Non-Substantive": 0
}

df["labels"] = df["Substantive_NonSubstantive"].map(label_mapping)

df = df[df["labels"].notna()]
df["labels"] = df["labels"].astype(int)

print("Count of 'S':", (df["Substantive_NonSubstantive"] == "S").sum())
print("Count of 'NS':", (df["Substantive_NonSubstantive"] == "NS").sum())
print("Count of 'Non-Substantive':", (df["Substantive_NonSubstantive"] == "Non-Substantive").sum())

df = df[["Comment", "labels"]]

df["Comment"] = df["Comment"].astype(str)

print("Final dataset size:", len(df))


dataset = dataset.cast_column(
    "labels",
    ClassLabel(num_classes=2, names=["NS", "S"])
)


tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["Comment"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["Comment", "__index_level_0__"])

train_test = dataset.train_test_split(
    test_size=0.2,
    stratify_by_column="labels"
)

train_dataset = train_test["train"]
test_dataset = train_test["test"]


accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision.compute(predictions=preds, references=labels)["precision"],
        "recall": recall.compute(predictions=preds, references=labels)["recall"],
        "f1": f1.compute(predictions=preds, references=labels)["f1"],
    }


def objective(trial):

    learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 8)

    training_args = TrainingArguments(
        output_dir=f"./optuna_runs/trial_{trial.number}",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=50,
        report_to="none",
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    eval_result = trainer.evaluate()
    return eval_result["eval_f1"]


print("⚡ Starting Optuna hyperparameter search...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
print("Best hyperparameters:", study.best_params)


best = study.best_params

training_args = TrainingArguments(
    output_dir="./comment_classifier",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=best["num_train_epochs"],
    learning_rate=best["learning_rate"],
    weight_decay=best["weight_decay"],
    warmup_ratio=best["warmup_ratio"],
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

eval_results = trainer.evaluate()

metrics_table = pd.DataFrame(
    [eval_results],
    columns=eval_results.keys()
)

print("\n Evaluation Metrics\n")
print(metrics_table.to_string(index=False))

trainer.save_model("./comment_classifier")
tokenizer.save_pretrained("./comment_classifier")

print(" Training complete. Final model saved to ./comment_classifier")