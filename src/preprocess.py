
import os
from data_pipeline import (
    create_or_load_shared_split,
    label_distribution,
    load_and_clean_labeled_dataset,
    save_clean_dataset,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "data", "dataset.csv")


if __name__ == "__main__":
    print("=" * 55)
    print("  Preprocessing -- Compiler Error Classifier")
    print("  Roll: 24CSB0A25")
    print("=" * 55)

    print("\nLoading dataset.csv ...")
    cleaned = load_and_clean_labeled_dataset(data_path)
    clean_path = save_clean_dataset(cleaned)
    split_bundle = create_or_load_shared_split(cleaned, test_size=0.2, random_state=42)

    print(f"  Cleaned labeled rows      : {len(cleaned)}")
    print(f"  Train/Test split          : {len(split_bundle.train_df)} / {len(split_bundle.test_df)}")
    print("  Label distribution (full cleaned dataset):")
    print(cleaned["label"].value_counts().to_string())
    print("  Label distribution (train):", label_distribution(split_bundle.train_df))
    print("  Label distribution (test) :", label_distribution(split_bundle.test_df))
    print(f"  Output -> {clean_path}")