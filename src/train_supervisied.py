import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, f1_score)
from data_pipeline import (
    create_or_load_shared_split,
    label_distribution,
    load_and_clean_labeled_dataset,
    save_clean_dataset,
)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "data", "processed", "clean_dataset.csv")
model_dir = os.path.join(BASE_DIR, "..", "models")
os.makedirs(model_dir, exist_ok=True)

print("=" * 55)
print("  Supervised Training -- Compiler Error Classifier")
print("  Roll: 24CSB0A25")
print("=" * 55)

print("\nLoading and validating cleaned dataset ...")
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    raw_path = os.path.join(BASE_DIR, "..", "data", "dataset.csv")
    df = load_and_clean_labeled_dataset(raw_path)
    save_clean_dataset(df, data_path)

print(f"  Total samples : {len(df)}")
print("  Label distribution:")
print(df["label"].value_counts().to_string())
split_bundle = create_or_load_shared_split(df, test_size=0.2, random_state=42)
train_df = split_bundle.train_df
test_df = split_bundle.test_df

X_train = train_df["error_message"].fillna("")
X_test = test_df["error_message"].fillna("")
y_train = train_df["label"]
y_test = test_df["label"]

print(f"\n  Train : {len(X_train)}  |  Test : {len(X_test)}")
print(f"  Train distribution : {label_distribution(train_df)}")
print(f"  Test distribution  : {label_distribution(test_df)}")

overlap = set(X_train) & set(X_test)
assert len(overlap) == 0, f"Leakage detected: {len(overlap)} overlapping texts"
print("  Train/Test overlap : 0 samples  [Clean]")

print("\nBuilding TF-IDF vectorizer ...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=4000,
    sublinear_tf=True,
    min_df=2,
    strip_accents='unicode',
    analyzer='word'
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
print(f"  Features : {X_train_vec.shape[1]}")

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, C=1.0, class_weight='balanced', solver='lbfgs'
    ),
    "Naive Bayes": MultinomialNB(alpha=1.0),
    "LinearSVC": LinearSVC(
        max_iter=2000, C=0.8, class_weight='balanced'
    ),
}

# Per-model pickle names in models/ (same TF-IDF for all; see tfidf_vectorizer.pkl)
MODEL_SAVE_FILES = {
    "Logistic Regression": "super_LR.pkl",
    "Naive Bayes": "super_NB.pkl",
    "LinearSVC": "super_SVC.pkl",
}

LABELS      = ["lexical", "syntax", "semantic"]
results     = {}
skf         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
report_lines = [
    "Compiler Error Classifier -- Accuracy Report\n",
    "Roll: 24CSB0A25\n",
    "=" * 50 + "\n\n",
    f"Dataset   : {len(df)} samples (after dedup)\n",
    f"Train set : {len(X_train)} samples\n",
    f"Test set  : {len(X_test)} samples\n",
    f"Features  : {X_train_vec.shape[1]} TF-IDF n-grams (1,2)\n\n",
]

for name, model in MODELS.items():
    print(f"\n{'='*55}\n  Training : {name}\n{'='*55}")

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc    = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average='macro')

    cv_scores = cross_val_score(
        model, X_train_vec, y_train,
        cv=skf, scoring='f1_macro', n_jobs=-1
    )

    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    cr = classification_report(y_test, y_pred, target_names=LABELS, digits=4)

    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  Test F1 Macro : {f1_mac*100:.2f}%")
    print(f"  CV F1 (macro) : {cv_scores.mean()*100:.2f}%  +/-  {cv_scores.std()*100:.2f}%")
    print(f"\n  Classification Report:\n{cr}")
    print(f"  Confusion Matrix  [lexical | syntax | semantic]")
    print(f"    lexical   {cm[0]}")
    print(f"    syntax    {cm[1]}")
    print(f"    semantic  {cm[2]}")

    results[name] = {
        "accuracy"   : acc,
        "f1_macro"   : f1_mac,
        "cv_f1_mean" : cv_scores.mean(),
        "cv_f1_std"  : cv_scores.std(),
        "model"      : model,
    }

    report_lines += [
        f"\n{name}\n" + "-"*40 + "\n",
        f"  Test Accuracy : {acc*100:.2f}%\n",
        f"  Test F1 Macro : {f1_mac*100:.2f}%\n",
        f"  CV F1 (macro) : {cv_scores.mean()*100:.2f}%  +/-  {cv_scores.std()*100:.2f}%\n\n",
        "  Classification Report:\n",
        cr + "\n",
        "  Confusion Matrix  [lexical | syntax | semantic]\n",
        f"    lexical   {cm[0]}\n",
        f"    syntax    {cm[1]}\n",
        f"    semantic  {cm[2]}\n",
    ]
    
print(f"\n{'='*55}\n  Model Comparison\n{'='*55}")
print(f"  {'Model':<25} {'Test Acc':>10}   {'Test F1':>8}   {'CV F1':>8}")
print("  " + "-"*55)
for name, r in results.items():
    print(f"  {name:<25} {r['accuracy']*100:>9.2f}%   "
          f"{r['f1_macro']*100:>7.2f}%   {r['cv_f1_mean']*100:>7.2f}%")

comparison = (
    "\n\nModel Comparison\n" + "="*50 + "\n"
    + f"  {'Model':<25} {'Test Acc':>10}   {'Test F1':>8}   {'CV F1':>8}\n"
    + "  " + "-"*55 + "\n"
)
for name, r in results.items():
    comparison += (
        f"  {name:<25} {r['accuracy']*100:>9.2f}%   "
        f"{r['f1_macro']*100:>7.2f}%   {r['cv_f1_mean']*100:>7.2f}%\n"
    )
report_lines.append(comparison)

best_name  = max(results, key=lambda k: results[k]["cv_f1_mean"])
best_model = results[best_name]["model"]

print(f"\n  Best Model : {best_name}")
print(f"  CV F1      : {results[best_name]['cv_f1_mean']*100:.2f}%")
report_lines.append(f"\nBest Model : {best_name}  (CV F1 = {results[best_name]['cv_f1_mean']*100:.2f}%)\n")

for train_name, fname in MODEL_SAVE_FILES.items():
    out_path = os.path.join(model_dir, fname)
    joblib.dump(results[train_name]["model"], out_path)
    print(f"  Saved {train_name} -> {out_path}")
report_lines.append(
    "\nPer-model artifacts (same TF-IDF; use tfidf_vectorizer.pkl):\n"
    + "\n".join(f"  {k} -> {v}" for k, v in MODEL_SAVE_FILES.items())
    + "\n"
)

joblib.dump(best_model, os.path.join(model_dir, "supervised_model.pkl"))
joblib.dump(vectorizer,  os.path.join(model_dir, "tfidf_vectorizer.pkl"))

report_path = os.path.join(model_dir, "accuracy_report.txt")
with open(report_path, "w") as f:
    f.writelines(report_lines)

print(f"\n  Best model (for predict.py) -> {model_dir}/supervised_model.pkl")
print(f"  Per-model PKL               -> super_LR.pkl, super_NB.pkl, super_SVC.pkl")
print(f"  Vectorizer saved -> {model_dir}/tfidf_vectorizer.pkl")
print(f"  Report saved     -> {report_path}")