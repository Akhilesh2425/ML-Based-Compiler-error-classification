import os
import pickle
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.preprocessing import label_binarize

from data_pipeline import (
    LABELS,
    create_or_load_shared_split,
    load_and_clean_labeled_dataset,
    save_clean_dataset,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def generate_multiclass_roc_curve() -> str:
    def load_test_dataset() -> Tuple[pd.Series, pd.Series]:
        clean_path = os.path.join(PROJECT_ROOT, "data", "processed", "clean_dataset.csv")
        if os.path.exists(clean_path):
            df = pd.read_csv(clean_path)
        else:
            raw_path = os.path.join(PROJECT_ROOT, "data", "dataset.csv")
            df = load_and_clean_labeled_dataset(raw_path)
            save_clean_dataset(df, clean_path)

        split_bundle = create_or_load_shared_split(df, test_size=0.2, random_state=42)
        return split_bundle.test_df["error_message"].fillna(""), split_bundle.test_df["label"]

    def first_existing(paths: List[str]) -> str:
        for path in paths:
            if os.path.exists(path):
                return path
        return ""

    def softmax(scores: np.ndarray) -> np.ndarray:
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def align_scores(scores: np.ndarray, classes: List[str]) -> np.ndarray:
        aligned = np.zeros((scores.shape[0], len(LABELS)))
        class_lookup = {str(label).lower(): idx for idx, label in enumerate(classes)}
        for out_idx, label in enumerate(LABELS):
            if label in class_lookup:
                aligned[:, out_idx] = scores[:, class_lookup[label]]
        return aligned

    def sklearn_scores(model, x_test_vec) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(x_test_vec)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(x_test_vec)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            scores = softmax(scores)
        else:
            raise ValueError("Selected model does not expose predict_proba() or decision_function().")

        classes = [str(label).lower() for label in getattr(model, "classes_", LABELS)]
        return align_scores(np.asarray(scores), classes)

    def bert_scores(x_test: pd.Series) -> Tuple[np.ndarray, List[str]]:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("BERT dependencies are not available.") from exc

        bert_dir = os.path.join(PROJECT_ROOT, "bert_model")
        if not os.path.isdir(bert_dir):
            raise FileNotFoundError("bert_model directory was not found.")

        bundle_path = first_existing(
            [
                os.path.join(PROJECT_ROOT, "output_models", "BERT_model.pkl"),
                os.path.join(PROJECT_ROOT, "output_models", "model.pkl"),
            ]
        )
        if bundle_path:
            with open(bundle_path, "rb") as f:
                pickle.load(f)

        tokenizer = AutoTokenizer.from_pretrained(bert_dir)
        model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
        model.eval()

        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        if set(id2label.values()).isdisjoint(set(LABELS)):
            id2label = {0: "lexical", 1: "syntax", 2: "semantic"}

        texts = [str(text) for text in x_test.tolist()]
        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), 64):
                batch_texts = texts[start : start + 64]
                encoded = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors="pt",
                )
                logits = model(**encoded).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        scores = np.vstack(all_probs)
        classes = [id2label[i] for i in range(scores.shape[1])]
        return align_scores(scores, classes), classes

    x_test, y_test = load_test_dataset()
    y_test_labels = [str(label).lower() for label in y_test.tolist()]

    chosen_name = ""
    y_score = None
    try:
        y_score, _ = bert_scores(x_test)
        chosen_name = "BERT"
    except Exception:
        vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        vectorizer = joblib.load(vectorizer_path)
        x_test_vec = vectorizer.transform(x_test)

        model_files: Dict[str, List[str]] = {
            "Logistic Regression": [
                os.path.join(MODELS_DIR, "lr_model.pkl"),
                os.path.join(MODELS_DIR, "super_LR.pkl"),
            ],
            "LinearSVC": [
                os.path.join(MODELS_DIR, "svc_model.pkl"),
                os.path.join(MODELS_DIR, "super_SVC.pkl"),
            ],
            "Naive Bayes": [
                os.path.join(MODELS_DIR, "nb_model.pkl"),
                os.path.join(MODELS_DIR, "super_NB.pkl"),
            ],
        }
        candidates = []
        for model_name, paths in model_files.items():
            model_path = first_existing(paths)
            if not model_path:
                continue
            model = joblib.load(model_path)
            y_pred = [str(label).lower() for label in model.predict(x_test_vec).tolist()]
            candidates.append((accuracy_score(y_test_labels, y_pred), model_name, model))

        if not candidates:
            raise RuntimeError("No usable trained model .pkl files were found.")

        _, chosen_name, best_model = max(candidates, key=lambda item: item[0])
        y_score = sklearn_scores(best_model, x_test_vec)

    y_true_bin = label_binarize(y_test_labels, classes=LABELS)

    fig, ax = plt.subplots(figsize=(8, 6))
    display_labels = {"lexical": "Lexical", "syntax": "Syntax", "semantic": "Semantic"}
    for class_idx, label in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_score[:, class_idx])
        roc_auc = auc(fpr, tpr)
        smooth_fpr = np.linspace(0, 1, 200)
        smooth_tpr = np.interp(smooth_fpr, fpr, tpr)
        smooth_tpr[0] = 0
        smooth_tpr[-1] = 1
        ax.plot(
            smooth_fpr,
            smooth_tpr,
            linewidth=2,
            label=f"{display_labels[label]} (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, label="Random")
    ax.set_title("Multiclass ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(title=chosen_name, loc="lower right")
    fig.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "multiclass_roc_curve.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_compilation_carbon_footprint() -> str:
    import shutil
    import subprocess
    import sys

    samples_dir = os.path.join(PROJECT_ROOT, "c_samples")
    sample_files = [
        os.path.join(samples_dir, filename)
        for filename in sorted(os.listdir(samples_dir))
        if filename.lower().endswith(".c")
    ]
    if not sample_files:
        raise RuntimeError("No C test cases found in c_samples/.")

    compiler = shutil.which("gcc") or shutil.which("clang")
    if not compiler:
        raise RuntimeError("No local C compiler found. Install gcc or clang to measure compilation carbon.")

    agentic_path = os.path.join(PROJECT_ROOT, "agentic_ai")
    if agentic_path not in sys.path:
        sys.path.insert(0, agentic_path)
    from green_metrics import GreenTracker

    run_count = 6
    runs = [f"Run {idx}" for idx in range(1, run_count + 1)]
    emissions = []
    for run_idx in range(1, run_count + 1):
        tracker = GreenTracker(prefer="estimated")
        tracker.start(label=f"compilation_run_{run_idx}")
        for sample_path in sample_files:
            subprocess.run(
                [compiler, "-fsyntax-only", "-Wall", sample_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        result = tracker.stop()
        emissions.append(result.co2_mg * 1_000)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        runs,
        emissions,
        color="#2dbb84",
        edgecolor="#1b4d3e",
        linewidth=1.5,
        width=0.6,
    )

    ax.set_title("Compilation Carbon Footprint per Test Case", fontsize=16, fontweight="bold")
    ax.set_ylabel("Carbon Emissions (ug CO2eq)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(emissions) * 1.18 if emissions else 1)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    for bar, value in zip(bars, emissions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.12,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "compilation_carbon_footprint.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(generate_multiclass_roc_curve())
