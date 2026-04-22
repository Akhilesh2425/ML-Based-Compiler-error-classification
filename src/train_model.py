import os
import pandas as pd
import pickle
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
def read_dataset(file_path):

    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        next(f)

        for line in f:
            line = line.strip()
            if line == "":
                continue

            try:
                msg, lbl = line.rsplit(",", 1)
            except ValueError:
                continue

            rows.append((msg.strip(), lbl.strip().lower()))

    data = pd.DataFrame(rows, columns=["error_message", "label"])

    valid_labels = {"lexical", "syntax", "semantic"}
    data = data[data["label"].isin(valid_labels)].reset_index(drop=True)

    print(f"[+] Loaded {len(data)} samples")
    print("Label distribution:")
    print(data["label"].value_counts(), "\n")

    return data

def clean_text(text):

    text = text.lower()
    text = re.sub(r"\\[0-9a-f]+", " ", text)
    text = re.sub(r"'[^']{1,10}'", " TOKEN ", text)
    text = re.sub(r"\b\d+\b", " NUM ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def build_features(train_text, test_text):

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=150,
        sublinear_tf=True
    )

    train_vec = tfidf.fit_transform(train_text)
    test_vec = tfidf.transform(test_text)

    print(f"[+] TF-IDF created with {train_vec.shape[1]} features\n")

    return train_vec, test_vec, tfidf

def mask_some_labels(y_train, ratio=0.4):

    rng = np.random.RandomState(42)

    y_masked = y_train.copy().to_numpy()

    count_unlabeled = int(len(y_masked) * ratio)
    idx = rng.choice(len(y_masked), count_unlabeled, replace=False)

    y_masked[idx] = -1

    labeled_count = np.sum(y_masked != -1)

    print("[+] Semi-supervised setup")
    print("Labeled   :", labeled_count)
    print("Unlabeled :", count_unlabeled)
    print("Total     :", len(y_masked), "\n")

    return y_masked

def train_models(train_vec, test_vec, y_train_semi, y_test):

    models = {
        "Semi NB": MultinomialNB(alpha=0.5),
        "Semi LR": LogisticRegression(max_iter=1000, solver="lbfgs")
    }

    results = {}
    trained_models = {}

    for name, base_model in models.items():

        clf = SelfTrainingClassifier(
            estimator=base_model,
            threshold=0.8,
            max_iter=10
        )

        clf.fit(train_vec, y_train_semi)

        preds = clf.predict(test_vec)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        results[name] = {"Accuracy": acc, "F1": f1}
        trained_models[name] = clf

        print(name)

        print("Accuracy :", f"{acc*100:.2f}%")
        print("Precision:", f"{prec*100:.2f}%")
        print("Recall   :", f"{rec*100:.2f}%")
        print("F1 Score :", f"{f1*100:.2f}%\n")

        print(classification_report(y_test, preds))

    print("Model comparison")

    for name, r in results.items():
        print(f"{name:20}  Acc: {r['Accuracy']*100:.2f}%  F1: {r['F1']*100:.2f}%")

    return results, trained_models

def save_best_model(results, trained_models, tfidf):

    best_name = max(results, key=lambda k: results[k]["Accuracy"])

    bundle = {
        "model": trained_models[best_name],
        "vectorizer": tfidf,
        "model_name": best_name
    }

    with open("compiler_error_classifier.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print("\nBest model saved:", best_name)
    print("Accuracy:", f"{results[best_name]['Accuracy']*100:.2f}%")

if __name__ == "__main__":
    try:
        print("Compiler Error Classification")
        print("=" * 45, "\n")

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        dataset_path = os.path.join(BASE_DIR, "..", "data", "processed", "clean_dataset.csv")

        print(f"[*] Loading data from: {dataset_path}")
        df = read_dataset(dataset_path)

        print("[*] Cleaning text...")
        df["clean"] = df["error_message"].apply(clean_text)

        print("[*] Dropping duplicates...")
        before_count = len(df)
        df = df.drop_duplicates(subset=["clean"]).reset_index(drop=True)
        print(f"[!] Dropped {before_count - len(df)} duplicates. Remaining: {len(df)}\n")

        print("[*] Splitting data...")
        train_text, test_text, y_train, y_test = train_test_split(
            df["clean"], df["label"], test_size=0.3, random_state=42, stratify=df["label"]
        )

        print("[*] Building TF-IDF features...")
        train_vec, test_vec, tfidf = build_features(train_text, test_text)

        print("[*] Masking labels for semi-supervised...")
        y_train_semi = mask_some_labels(y_train, ratio=0.4)

        print("[*] Training models ")
        results, trained_models = train_models(train_vec, test_vec, y_train_semi, y_test)

        save_best_model(results, trained_models, tfidf)
        print("\n[+] Training completed successfully!")

    except Exception as e:
        import traceback
        print("\n" + "!" * 45)
        print("🚨 CRASH DETECTED 🚨")
        print("!" * 45)
        print(traceback.format_exc())