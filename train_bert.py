import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import inspect
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from datasets import Dataset
from src.data_pipeline import (
    LABELS,
    create_or_load_shared_split,
    label_distribution,
    load_and_clean_labeled_dataset,
    save_clean_dataset,
)

# =========================
# 1. LOAD CLEANED DATA + SHARED SPLIT
# =========================
clean_dataset_path = os.path.join("data", "processed", "clean_dataset.csv")
if os.path.exists(clean_dataset_path):
    df = pd.read_csv(clean_dataset_path)
else:
    raw_dataset_path = os.path.join("data", "dataset.csv")
    df = load_and_clean_labeled_dataset(raw_dataset_path)
    save_clean_dataset(df, clean_dataset_path)

split_bundle = create_or_load_shared_split(df, test_size=0.2, random_state=42)
train_full_df = split_bundle.train_df.copy()
test_df = split_bundle.test_df.copy()

# Validation split ONLY from training portion.
train_df, val_df = train_test_split(
    train_full_df,
    test_size=0.1,
    random_state=42,
    stratify=train_full_df["label"],
)

label2id = {label: idx for idx, label in enumerate(LABELS)}
id2label = {idx: label for label, idx in label2id.items()}

for frame in (train_df, val_df, test_df):
    frame["label"] = frame["label"].astype(str).str.lower()
    frame["label_id"] = frame["label"].map(label2id).astype(int)

print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))
print("Train distribution:", label_distribution(train_df))
print("Val distribution  :", label_distribution(val_df))
print("Test distribution :", label_distribution(test_df))

# =========================
# 4. LOAD MODEL + TOKENIZER
# =========================
model_name = "distilbert-base-uncased"   # faster + stable

tokenizer = AutoTokenizer.from_pretrained(model_name)

num_labels = len(LABELS)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    dropout=0.2,
    attention_dropout=0.2,
    seq_classif_dropout=0.3,
)

# =========================
# 5. CONVERT TO DATASET
# =========================
def convert_to_dataset(data):
    return Dataset.from_pandas(
        data[["error_message", "label_id"]]
        .rename(columns={"label_id": "labels"}),
        preserve_index=False
    )

train_dataset = convert_to_dataset(train_df)
val_dataset = convert_to_dataset(val_df)
test_dataset = convert_to_dataset(test_df)

# =========================
# 6. TOKENIZATION
# =========================
def tokenize(example):
    return tokenizer(
        example["error_message"],
        truncation=True,
        padding=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }

# =========================
# 8. TRAINING CONFIG (5.x FIX)
# =========================
training_kwargs = {
    "output_dir": "./bert_output",
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 3,
    "weight_decay": 0.01,
    "logging_steps": 50,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1_macro",
    "greater_is_better": True,
}

training_args_params = inspect.signature(TrainingArguments.__init__).parameters

# Transformers versions use either "evaluation_strategy" or "eval_strategy".
if "evaluation_strategy" in training_args_params:
    training_kwargs["evaluation_strategy"] = "epoch"
elif "eval_strategy" in training_args_params:
    training_kwargs["eval_strategy"] = "epoch"

# Disable external logging integrations.
if "report_to" in training_args_params:
    training_kwargs["report_to"] = "none"

# Force CPU usage in a version-compatible way.
if "use_cpu" in training_args_params:
    training_kwargs["use_cpu"] = True
elif "no_cuda" in training_args_params:
    training_kwargs["no_cuda"] = True

# Optional quick test run: set FAST_DEV_STEPS env var (e.g. 10)
fast_dev_steps = os.getenv("FAST_DEV_STEPS")
if fast_dev_steps:
    training_kwargs["max_steps"] = int(fast_dev_steps)

training_args = TrainingArguments(**training_kwargs)

# =========================
# 9. TRAINER
# =========================
trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": train_dataset,
    "eval_dataset": val_dataset,
    "compute_metrics": compute_metrics,
}

trainer_params = inspect.signature(Trainer.__init__).parameters
if "tokenizer" in trainer_params:
    trainer_kwargs["tokenizer"] = tokenizer
elif "processing_class" in trainer_params:
    trainer_kwargs["processing_class"] = tokenizer

trainer = Trainer(**trainer_kwargs)
trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=1))

# =========================
# 10. TRAIN
# =========================
steps_per_epoch = (len(train_dataset) + training_kwargs["per_device_train_batch_size"] - 1) // training_kwargs["per_device_train_batch_size"]
print(f"\nTraining started... (about {steps_per_epoch} steps/epoch)\n")
trainer.train()

# =========================
# 11. EVALUATE
# =========================
print("\nValidation Results:")
print(trainer.evaluate(val_dataset))

print("\nTest Results:")
test_metrics = trainer.evaluate(test_dataset)
print(test_metrics)

pred_output = trainer.predict(test_dataset)
test_preds = np.argmax(pred_output.predictions, axis=1)
true_ids = pred_output.label_ids
true_labels = [id2label[int(x)] for x in true_ids]
pred_labels = [id2label[int(x)] for x in test_preds]

print("\nClassification Report (Test):")
print(classification_report(true_labels, pred_labels, labels=LABELS, target_names=LABELS, digits=4))
print("Confusion Matrix [lexical | syntax | semantic]:")
print(confusion_matrix(true_labels, pred_labels, labels=LABELS))

# =========================
# 12. SAVE MODEL
# =========================
trainer.save_model("bert_model") 
tokenizer.save_pretrained("bert_model")

# Save a pickle sidecar for projects expecting a .pkl artifact.
os.makedirs("output_models", exist_ok=True)
pkl_bundle = {
    "model_dir": os.path.abspath("bert_model"),
    "tokenizer_dir": os.path.abspath("bert_model"),
    "id2label": id2label,
    "label2id": label2id,
    "num_labels": num_labels,
    "model_name": model_name,
}
with open(os.path.join("output_models", "model.pkl"), "wb") as f:
    pickle.dump(pkl_bundle, f)

print("\nTraining Complete!")
print(f"PKL metadata saved to: {os.path.abspath(os.path.join('output_models', 'model.pkl'))}")