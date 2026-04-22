

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def create_tfidf_vectorizer(text_data, save_path: str):
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
        min_df=1,
        strip_accents='unicode',
        analyzer='word'
    )
    X_vec = vectorizer.fit_transform(text_data)
    joblib.dump(vectorizer, save_path)
    print(f"[features] TF-IDF vectorizer saved → {save_path}")
    print(f"[features] Features : {X_vec.shape[1]}")
    return X_vec


def create_count_vectorizer(text_data, save_path: str):
    """Fit a Count vectorizer, save it, return the matrix."""
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        min_df=1
    )
    X_vec = vectorizer.fit_transform(text_data)
    joblib.dump(vectorizer, save_path)
    print(f"[features] Count vectorizer saved → {save_path}")
    return X_vec


def load_vectorizer(path: str):
    """Load a previously saved vectorizer from disk."""
    vec = joblib.load(path)
    print(f"[features] Vectorizer loaded ← {path}")
    return vec