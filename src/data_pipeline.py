import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


LABELS = ["lexical", "syntax", "semantic"]
VALID_LABELS = set(LABELS)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SPLIT_PATH = os.path.join(PROCESSED_DIR, "shared_split.json")
CLEANED_PATH = os.path.join(PROCESSED_DIR, "clean_dataset.csv")


@dataclass
class SplitBundle:
    df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_indices: list[int]
    test_indices: list[int]


def normalize_error_message(msg: str) -> str:
    msg = str(msg).lower()
    msg = re.sub(r"\r\n?", "\n", msg)
    msg = re.sub(r"^(error|warning|note)\s*:\s*", "", msg)
    msg = re.sub(r"\bline\s+\d+\b", " line_num ", msg)
    msg = re.sub(r"\bcolumn\s+\d+\b", " column_num ", msg)
    msg = re.sub(r"\b\d+\b", " num ", msg)
    msg = re.sub(r"[a-zA-Z]:\\[^\s]+", " path ", msg)
    msg = re.sub(r"/\S+", " path ", msg)
    msg = re.sub(r"`[^`]+`", " token ", msg)
    msg = re.sub(r"'[^']+'", " token ", msg)
    msg = re.sub(r"\"[^\"]+\"", " token ", msg)
    msg = re.sub(r"\b(tokenization|lexer|lexical)\b", " stage ", msg)
    msg = re.sub(r"\b(parser|parsing|syntax)\b", " stage ", msg)
    msg = re.sub(r"\b(type/check|typecheck|semantic)\b", " stage ", msg)
    msg = re.sub(r"\b(stage\s+note|parser\s+note|tokenization\s+notice)\b", " stage_note ", msg)
    msg = re.sub(r"\s+", " ", msg)
    msg = re.sub(r"[^a-z0-9_ ]", " ", msg)
    return re.sub(r"\s+", " ", msg).strip()


def _near_duplicate_key(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip())
    short = " ".join(collapsed.split()[:24])
    return hashlib.md5(short.encode("utf-8")).hexdigest()


def load_and_clean_labeled_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    if "error_message" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'error_message' and 'label' columns.")

    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(VALID_LABELS)].copy()
    df["error_message"] = df["error_message"].astype(str).apply(normalize_error_message)
    df = df[df["error_message"].str.len() > 3].copy()

    df["near_key"] = df["error_message"].apply(_near_duplicate_key)
    df = df.drop_duplicates(subset=["error_message", "label"])
    df = df.drop_duplicates(subset=["near_key", "label"])
    df = df.drop(columns=["near_key"])

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def _assert_no_text_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    overlap = set(train_df["error_message"]) & set(test_df["error_message"])
    assert len(overlap) == 0, f"Leakage detected: {len(overlap)} overlapping messages"


def create_or_load_shared_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> SplitBundle:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    needs_new_split = True
    if os.path.exists(SPLIT_PATH):
        with open(SPLIT_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        saved_size = int(saved.get("dataset_size", -1))
        if saved_size == len(df):
            train_idx = [int(x) for x in saved["train_indices"]]
            test_idx = [int(x) for x in saved["test_indices"]]
            needs_new_split = False

    if needs_new_split:
        indices = list(range(len(df)))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=df["label"],
        )
        with open(SPLIT_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "random_state": random_state,
                    "test_size": test_size,
                    "dataset_size": len(df),
                    "train_indices": train_idx,
                    "test_indices": test_idx,
                },
                f,
                indent=2,
            )

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    assert len(train_df) + len(test_df) == len(df), "Split size mismatch"
    _assert_no_text_overlap(train_df, test_df)
    return SplitBundle(
        df=df,
        train_df=train_df,
        test_df=test_df,
        train_indices=train_idx,
        test_indices=test_idx,
    )


def save_clean_dataset(df: pd.DataFrame, target_path: str = CLEANED_PATH) -> str:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    df.to_csv(target_path, index=False)
    return target_path


def label_distribution(df: pd.DataFrame) -> Dict[str, int]:
    return {label: int((df["label"] == label).sum()) for label in LABELS}
