import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from data_pipeline import (
    LABELS,
    create_or_load_shared_split,
    label_distribution,
    load_and_clean_labeled_dataset,
    save_clean_dataset,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORT_PATH = os.path.join(MODELS_DIR, "accuracy_report.txt")


@dataclass
class ModelResult:
    name: str
    source: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    cv_f1_mean: float | None
    cv_f1_std: float | None
    y_true: List[str]
    y_pred: List[str]
    class_report: str
    conf_matrix: np.ndarray
    extra_note: str = ""


def _mod_time(path: str) -> float | None:
    if not os.path.exists(path):
        return None
    return os.path.getmtime(path)


def _format_mtime(ts: float | None) -> str:
    if ts is None:
        return "missing"
    return pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d %H:%M:%S")


def _build_freshness_notes() -> Tuple[List[str], List[str]]:
    clean_dataset = os.path.join(PROJECT_ROOT, "data", "processed", "clean_dataset.csv")
    supervised_artifacts = [
        os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"),
        os.path.join(MODELS_DIR, "super_LR.pkl"),
        os.path.join(MODELS_DIR, "super_NB.pkl"),
        os.path.join(MODELS_DIR, "super_SVC.pkl"),
        os.path.join(MODELS_DIR, "supervised_model.pkl"),
    ]
    bert_artifact = os.path.join(PROJECT_ROOT, "bert_model", "config.json")

    clean_ts = _mod_time(clean_dataset)
    supervised_latest_ts = max((ts for ts in (_mod_time(p) for p in supervised_artifacts) if ts is not None), default=None)
    bert_model_ts = _mod_time(bert_artifact)

    lines = [
        "Freshness Check:",
        f"  clean_dataset.csv         : {_format_mtime(clean_ts)}",
        f"  Supervised model artifacts: {_format_mtime(supervised_latest_ts)}",
        f"  bert_model/config.json    : {_format_mtime(bert_model_ts)}",
        "",
    ]

    warnings: List[str] = []
    if clean_ts is not None and supervised_latest_ts is not None and clean_ts > supervised_latest_ts:
        warnings.append(
            "  - clean_dataset.csv is newer than supervised model artifacts. "
            "Re-run preprocessing/training before trusting classical metrics."
        )
    return lines, warnings


def _load_supervised_dataset() -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "clean_dataset.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        raw_path = os.path.join(PROJECT_ROOT, "data", "dataset.csv")
        df = load_and_clean_labeled_dataset(raw_path)
        save_clean_dataset(df, data_path)
    split_bundle = create_or_load_shared_split(df, test_size=0.2, random_state=42)
    x_train = split_bundle.train_df["error_message"].fillna("")
    x_test = split_bundle.test_df["error_message"].fillna("")
    y_train = split_bundle.train_df["label"]
    y_test = split_bundle.test_df["label"]
    return df, x_train, x_test, y_train, y_test


def _load_tfidf():
    return joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))


def _compute_core_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def _evaluate_sklearn_model(
    model_name: str,
    model_file: str,
    x_train_vec,
    x_test_vec,
    y_train: pd.Series,
    y_test: pd.Series,
    include_cv: bool = True,
    note: str = "",
) -> ModelResult:
    model = joblib.load(os.path.join(MODELS_DIR, model_file))
    y_pred = model.predict(x_test_vec)
    metrics = _compute_core_metrics(y_test.tolist(), y_pred.tolist())

    cv_mean = None
    cv_std = None
    if include_cv:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, x_train_vec, y_train, cv=skf, scoring="f1_macro", n_jobs=-1
        )
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

    class_rep = classification_report(
        y_test, y_pred, labels=LABELS, target_names=LABELS, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    return ModelResult(
        name=model_name,
        source=model_file,
        accuracy=metrics["accuracy"],
        precision_macro=metrics["precision_macro"],
        recall_macro=metrics["recall_macro"],
        f1_macro=metrics["f1_macro"],
        precision_weighted=metrics["precision_weighted"],
        recall_weighted=metrics["recall_weighted"],
        f1_weighted=metrics["f1_weighted"],
        cv_f1_mean=cv_mean,
        cv_f1_std=cv_std,
        y_true=y_test.tolist(),
        y_pred=y_pred.tolist(),
        class_report=class_rep,
        conf_matrix=cm,
        extra_note=note,
    )


def _evaluate_bert_model(test_texts: List[str], test_labels: List[str]) -> ModelResult:
    bert_model_dir = os.path.join(PROJECT_ROOT, "bert_model")
    true_labels = [str(x).lower() for x in test_labels]
    texts = [str(x) for x in test_texts]

    tokenizer = AutoTokenizer.from_pretrained(bert_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(bert_model_dir)
    model.eval()
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    if set(id2label.values()).isdisjoint(set(LABELS)):
        id2label = {0: "lexical", 1: "syntax", 2: "semantic"}

    batch_size = 64
    pred_labels: List[str] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_text = texts[i : i + batch_size]
            enc = tokenizer(
                batch_text,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            )
            logits = model(**enc).logits
            pred_ids = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            pred_labels.extend([str(id2label.get(int(idx), "unknown")).lower() for idx in pred_ids])

    if any(label not in LABELS for label in pred_labels):
        mapped = []
        for label in pred_labels:
            if label in LABELS:
                mapped.append(label)
            elif label.isdigit() and int(label) in (0, 1, 2):
                mapped.append(LABELS[int(label)])
            else:
                mapped.append("lexical")
        pred_labels = mapped

    metrics = _compute_core_metrics(true_labels, pred_labels)
    class_rep = classification_report(
        true_labels, pred_labels, labels=LABELS, target_names=LABELS, digits=4, zero_division=0
    )
    cm = confusion_matrix(true_labels, pred_labels, labels=LABELS)

    note = f"Evaluated on shared test split ({len(true_labels)} samples)."
    return ModelResult(
        name="BERT (DistilBERT)",
        source="bert_model/",
        accuracy=metrics["accuracy"],
        precision_macro=metrics["precision_macro"],
        recall_macro=metrics["recall_macro"],
        f1_macro=metrics["f1_macro"],
        precision_weighted=metrics["precision_weighted"],
        recall_weighted=metrics["recall_weighted"],
        f1_weighted=metrics["f1_weighted"],
        cv_f1_mean=None,
        cv_f1_std=None,
        y_true=true_labels,
        y_pred=pred_labels,
        class_report=class_rep,
        conf_matrix=cm,
        extra_note=note,
    )


def _format_result_block(res: ModelResult) -> str:
    lines = [
        f"\n{res.name}",
        "-" * 60,
        f"  Artifact           : {res.source}",
        f"  Test Accuracy      : {res.accuracy * 100:.2f}%",
        f"  Test Precision Mac : {res.precision_macro * 100:.2f}%",
        f"  Test Recall Mac    : {res.recall_macro * 100:.2f}%",
        f"  Test F1 Macro      : {res.f1_macro * 100:.2f}%",
        f"  Test Precision Wgt : {res.precision_weighted * 100:.2f}%",
        f"  Test Recall Wgt    : {res.recall_weighted * 100:.2f}%",
        f"  Test F1 Weighted   : {res.f1_weighted * 100:.2f}%",
    ]

    if res.cv_f1_mean is not None and res.cv_f1_std is not None:
        lines.append(f"  CV F1 (macro)      : {res.cv_f1_mean * 100:.2f}%  +/-  {res.cv_f1_std * 100:.2f}%")
    else:
        lines.append("  CV F1 (macro)      : N/A")

    if res.extra_note:
        lines.append(f"  Note               : {res.extra_note}")

    lines += [
        "",
        "  Classification Report:",
        res.class_report.rstrip(),
        "",
        "  Confusion Matrix  [lexical | syntax | semantic]",
        f"    lexical   {res.conf_matrix[0]}",
        f"    syntax    {res.conf_matrix[1]}",
        f"    semantic  {res.conf_matrix[2]}",
    ]
    return "\n".join(lines) + "\n"


def _is_supervised_duplicate(
    supervised_result: ModelResult, top_classical_result: ModelResult
) -> bool:
    same_predictions = supervised_result.y_pred == top_classical_result.y_pred
    same_scores = (
        abs(supervised_result.f1_macro - top_classical_result.f1_macro) < 1e-12
        and abs(supervised_result.accuracy - top_classical_result.accuracy) < 1e-12
    )
    return same_predictions and same_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model metrics report.")
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write report to file. By default, only prints to console.",
    )
    parser.add_argument(
        "--report-path",
        default=REPORT_PATH,
        help="Target report file path when --write-report is used.",
    )
    parser.add_argument(
        "--print-full-report",
        action="store_true",
        help="Print full report text to console.",
    )
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Loading model artifacts and computing metrics...")

    df, x_train, x_test, y_train, y_test = _load_supervised_dataset()
    tfidf = _load_tfidf()
    x_train_vec = tfidf.transform(x_train)
    x_test_vec = tfidf.transform(x_test)

    lr = _evaluate_sklearn_model("Logistic Regression", "super_LR.pkl", x_train_vec, x_test_vec, y_train, y_test)
    nb = _evaluate_sklearn_model("Naive Bayes", "super_NB.pkl", x_train_vec, x_test_vec, y_train, y_test)
    svc = _evaluate_sklearn_model("LinearSVC", "super_SVC.pkl", x_train_vec, x_test_vec, y_train, y_test)
    supervised = _evaluate_sklearn_model(
        "Supervised Bundle Model", "supervised_model.pkl", x_train_vec, x_test_vec, y_train, y_test
    )
    bert = _evaluate_bert_model(x_test.tolist(), y_test.tolist())

    classical_results = [lr, nb, svc]
    best_classical = max(classical_results, key=lambda r: (r.f1_macro, r.accuracy))
    skip_supervised = _is_supervised_duplicate(supervised, best_classical)

    all_results = classical_results + [bert]
    if not skip_supervised:
        all_results.insert(3, supervised)

    report_lines = [
        "Compiler Error Classifier -- Updated Metrics Report",
        "Roll: 24CSB0A25",
        "=" * 60,
        "",
        f"Supervised dataset   : {len(df)} samples (clean_dataset.csv)",
        f"Train/Test split     : {len(x_train)} / {len(x_test)} (random_state=42, stratified)",
        f"TF-IDF features      : {x_train_vec.shape[1]}",
        "",
        "Important Note:",
        "  All models are evaluated on the same shared 80/20 stratified split",
        "  from clean_dataset.csv (random_state=42).",
        "",
        f"Train label distribution: {label_distribution(pd.DataFrame({'label': y_train}))}",
        f"Test label distribution : {label_distribution(pd.DataFrame({'label': y_test}))}",
        "",
    ]
    freshness_lines, freshness_warnings = _build_freshness_notes()
    report_lines.extend(freshness_lines)
    if freshness_warnings:
        report_lines.append("Freshness Warnings:")
        report_lines.extend(freshness_warnings)
        report_lines.append("")

    if skip_supervised:
        report_lines.extend(
            [
                "Supervised Model Check:",
                "  supervised_model.pkl is identical in predictions and scores",
                f"  to best classical model: {best_classical.name}.",
                "  -> It is excluded from detailed comparison.",
                "",
            ]
        )
    else:
        report_lines.extend(
            [
                "Supervised Model Check:",
                f"  supervised_model.pkl differs from best classical model ({best_classical.name}).",
                "  -> It is included in detailed comparison.",
                "",
            ]
        )

    for res in all_results:
        report_lines.append(_format_result_block(res))

    report_lines.extend(
        [
            "",
            "Model Comparison",
            "=" * 60,
            f"  {'Model':<25} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12} {'CV F1':>10}",
            "  " + "-" * 60,
        ]
    )
    for res in all_results:
        cv_text = f"{res.cv_f1_mean * 100:.2f}%" if res.cv_f1_mean is not None else "N/A"
        report_lines.append(
            f"  {res.name:<25} {res.accuracy * 100:>9.2f}% {res.f1_macro * 100:>9.2f}%"
            f" {res.f1_weighted * 100:>11.2f}% {cv_text:>10}"
        )

    overall_best = max(all_results, key=lambda r: (r.f1_macro, r.accuracy))
    report_lines.extend(
        [
            "",
            f"Best Model by Test F1 Macro: {overall_best.name} ({overall_best.f1_macro * 100:.2f}%)",
            "",
            "Artifacts considered:",
            "  super_LR.pkl, super_NB.pkl, super_SVC.pkl, supervised_model.pkl, bert_model/",
        ]
    )

    report_text = "\n".join(report_lines).rstrip() + "\n"
    if args.print_full_report:
        print(report_text)

    if args.write_report:
        with open(args.report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Results are written to: {args.report_path}")
    else:
        print(f"Results are available at: {args.report_path}")


if __name__ == "__main__":
    main()
