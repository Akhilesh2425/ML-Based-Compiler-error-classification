import pandas as pd
import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "data", "processed", "clean_dataset.csv")
model_dir = os.path.join(BASE_DIR, "..", "models")


print("Semi-Supervised Training -> Self-Training")
print("=" * 55)
df = pd.read_csv(data_path)
df["label"] = df["label"].astype(str).str.strip().str.lower()

VALID_LABELS = {"lexical", "syntax", "semantic"}

labeled   = df[df["label"].isin(VALID_LABELS)].copy()
unlabeled = df[~df["label"].isin(VALID_LABELS)].copy()

print(f"\nLabeled samples   : {len(labeled)}")
print(f"Unlabeled samples : {len(unlabeled)}  <- intentionally blank in dataset")

if len(unlabeled) == 0:
    print("\n No unlabeled rows found.")
    print("   Ensure preprocess.py was run — it keeps the 144 unlabeled rows.")
    exit(1)

X = labeled["error_message"].fillna("")
y = labeled["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")

if not os.path.exists(tfidf_path):
    print("\n ERROR: tfidf_vectorizer.pkl not found!")
    print("   Run train_supervised.py first to create it.")
    exit(1)

tfidf_vec = joblib.load(tfidf_path)
print(f"\nLoaded vectorizer ({tfidf_vec.max_features} features)")

X_train_vec = tfidf_vec.transform(X_train)
X_test_vec  = tfidf_vec.transform(X_test)

print("\nTraining initial supervised baseline...")

lr = LogisticRegression(
    max_iter=2000,
    C=2,
    class_weight='balanced',
    solver='lbfgs',
)
lr.fit(X_train_vec, y_train)

baseline_preds = lr.predict(X_test_vec)
baseline_acc   = accuracy_score(y_test, baseline_preds)
print(f"Baseline Accuracy : {baseline_acc*100:.2f}%")

THRESHOLD     = 0.85
MAX_ROUNDS    = 5 
MAX_PER_ROUND = 50  


X_unl_all = unlabeled["error_message"].fillna("")
X_unl_vec = tfidf_vec.transform(X_unl_all)

# Working copies of training set
train_msgs_ext   = list(X_train)
train_labels_ext = list(y_train)
remaining_mask   = np.ones(len(X_unl_all), dtype=bool)  # True = still unlabeled

print(f"\nSelf-training on {len(unlabeled)} unlabeled samples")
print(f"(threshold={THRESHOLD}, max_rounds={MAX_ROUNDS}, max_per_round={MAX_PER_ROUND})")
print("-" * 55)

for rnd in range(1, MAX_ROUNDS + 1):

    remaining_idx = np.where(remaining_mask)[0]
    if len(remaining_idx) == 0:
        print(f"Round {rnd}: All unlabeled samples have been labeled. Done.")
        break

    X_remaining = X_unl_vec[remaining_idx]
    probs        = lr.predict_proba(X_remaining)
    max_probs    = probs.max(axis=1)
    pred_labels  = lr.classes_[probs.argmax(axis=1)]


    confident = np.where(max_probs >= THRESHOLD)[0]

    if len(confident) == 0:
        print(f"Round {rnd}: No predictions above threshold {THRESHOLD}. Stopping.")
        break


    if len(confident) > MAX_PER_ROUND:
        top_idx   = np.argsort(max_probs[confident])[::-1][:MAX_PER_ROUND]
        confident = confident[top_idx]


    chosen_idx    = remaining_idx[confident]
    pseudo_msgs   = X_unl_all.iloc[chosen_idx].tolist()
    pseudo_labels = pred_labels[confident].tolist()


    from collections import Counter
    label_counts = Counter(pseudo_labels)

    train_msgs_ext   += pseudo_msgs
    train_labels_ext += pseudo_labels
    remaining_mask[chosen_idx] = False  # mark as labeled

    # Retrain on expanded set
    X_ext_vec = tfidf_vec.transform(train_msgs_ext)
    lr.fit(X_ext_vec, train_labels_ext)

    round_preds = lr.predict(X_test_vec)
    round_acc   = accuracy_score(y_test, round_preds)

    still_unlabeled = remaining_mask.sum()
    print(f"Round {rnd}: +{len(confident)} pseudo-labels {dict(label_counts)} | "
          f"Still unlabeled: {still_unlabeled} | Test Acc: {round_acc*100:.2f}%")

final_model = lr

print("\nFinal pseudo-labels assigned to unlabeled samples:")
all_unl_vec    = tfidf_vec.transform(X_unl_all)
final_preds_unl = final_model.predict(all_unl_vec)
final_probs_unl = final_model.predict_proba(all_unl_vec).max(axis=1)
print(f"{'Error Message':<55} {'Label':<10} {'Conf':>6}")
print("-" * 75)
for msg, lbl, conf in zip(X_unl_all, final_preds_unl, final_probs_unl):
    print(f"{str(msg)[:54]:<55} {lbl:<10} {conf*100:>5.1f}%")


final_preds = final_model.predict(X_test_vec)
final_acc   = accuracy_score(y_test, final_preds)

print(f"\n{'='*55}")
print("Final Semi-Supervised Model Results")
print("="*55)
print(f"Baseline Accuracy : {baseline_acc*100:.2f}%")
print(f"Final Accuracy    : {final_acc*100:.2f}%")
print(f"Improvement       : {(final_acc - baseline_acc)*100:+.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, final_preds))

save_path = os.path.join(model_dir, "semi_supervised_model.pkl")
joblib.dump(final_model, save_path)
print(f" Semi-supervised model saved → {save_path}")