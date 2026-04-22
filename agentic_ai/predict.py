import os
import re
import pickle
import logging
import tempfile
import subprocess
from collections import Counter
from typing import Dict, List, Optional

import joblib
import numpy as np
import scipy.sparse as sp
logger = logging.getLogger(__name__)



# ---------------- AST SETUP ---------------- #
try:
    from pycparser import parse_file, c_ast
    PYCPARSER_AVAILABLE = True
except ImportError:
    PYCPARSER_AVAILABLE = False


class ASTVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.nodes = []

    def generic_visit(self, node):
        self.nodes.append(type(node).__name__)
        super().generic_visit(node)


# ---------------- PATHS ---------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))

FAKE_LIBC_PATH = os.path.join(_HERE, "fake_libc_include")
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_BERT_BUNDLE_PATH = os.path.join(_ROOT, "output_models", "model.pkl")
_BERT_MODEL_DIR_FALLBACK = os.path.join(_ROOT, "bert_model")

_PATHS = {
    "supervised_model": os.path.join(_HERE, "supervised_model.pkl"),
    "supervised_vectorizer": os.path.join(_HERE, "supervised_tfidf_vectorizer.pkl"),
    "svm_classifier": os.path.join(_HERE, "svm_classifier.pkl"),
    "svm_vectorizer": os.path.join(_HERE, "tfidf_vectorizer.pkl"),
    "ast_columns": os.path.join(_HERE, "ast_columns.pkl"),
    "stacking_model": os.path.join(_ROOT, "models", "stacking_model.pkl"),
    "stacking_vectorizer": os.path.join(_ROOT, "models", "tfidf_vectorizer.pkl"),
}


# ---------------- MODEL REGISTRY ---------------- #
class ModelRegistry:
    def __init__(self):
        self.cache = {}

    def load(self, key):
        if key in self.cache:
            return self.cache[key]

        path = _PATHS[key]
        if not os.path.exists(path):
            # Avoid emoji in console logs (Windows cp1252 can crash on them)
            print(f"[MISSING] {path}")
            self.cache[key] = None
            return None

        try:
            obj = joblib.load(path)
            self.cache[key] = obj
            print(f"[LOADED] {key}")
            return obj
        except Exception as e:
            print(f"[FAILED] {key}: {e}")
            self.cache[key] = None
            return None

    def supervised_available(self):
        return self.load("supervised_model") is not None and self.load("supervised_vectorizer") is not None

    def svm_available(self):
        return self.load("svm_classifier") is not None and self.load("svm_vectorizer") is not None

    def stacking_available(self):
        return self.load("stacking_model") is not None and self.load("stacking_vectorizer") is not None


_registry = ModelRegistry()

# BERT runtime (lazy): matches artifacts from ../train_bert.py → output_models/model.pkl + bert_model/
_bert_runtime: Dict[str, Optional[object]] = {
    "bundle": None,
    "model": None,
    "tokenizer": None,
    "model_dir": None,
}


def _load_bert_bundle_dict() -> Optional[Dict]:
    path = _BERT_BUNDLE_PATH
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning("BERT bundle load failed: %s", e)
        return None


def _resolve_bert_model_dir(bundle: Optional[Dict]) -> Optional[str]:
    if not bundle:
        return None
    md = bundle.get("model_dir")
    if md and os.path.isdir(md):
        return md
    fallback = _BERT_MODEL_DIR_FALLBACK
    if os.path.isdir(fallback):
        return fallback
    return None


def bert_available() -> bool:
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: F401
    except ImportError:
        return False
    bundle = _load_bert_bundle_dict()
    if not bundle:
        return False
    return _resolve_bert_model_dir(bundle) is not None


def _ensure_bert_loaded():
    """Load Hugging Face model/tokenizer once; uses bundle from train_bert.py."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if _bert_runtime.get("model") is not None and _bert_runtime.get("tokenizer") is not None:
        return _bert_runtime["model"], _bert_runtime["tokenizer"], _bert_runtime["bundle"]

    bundle = _load_bert_bundle_dict()
    if not bundle:
        raise RuntimeError(f"BERT metadata missing: {_BERT_BUNDLE_PATH}")
    model_dir = _resolve_bert_model_dir(bundle)
    if not model_dir:
        raise RuntimeError(f"BERT weights folder not found (expected {_BERT_MODEL_DIR_FALLBACK})")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    _bert_runtime["bundle"] = bundle
    _bert_runtime["model"] = model
    _bert_runtime["tokenizer"] = tokenizer
    _bert_runtime["model_dir"] = model_dir
    return model, tokenizer, bundle


def _predict_bert(text: str) -> Dict:
    import torch

    model, tokenizer, bundle = _ensure_bert_loaded()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    id2label = bundle.get("id2label") or {}
    raw_label = id2label.get(top_idx, id2label.get(str(top_idx), str(top_idx)))
    confidence = round(float(probs[top_idx]) * 100.0, 2)
    base_name = bundle.get("model_name") or "transformers"
    return {
        "label": str(raw_label),
        "confidence": confidence,
        "model_used": f"BERT ({base_name})",
    }


def get_model_status() -> Dict[str, bool]:
    """Which predictors can run (for Streamlit sidebar)."""
    return {
        "supervised": _registry.supervised_available(),
        "svm": _registry.svm_available(),
        "stacking": _registry.stacking_available(),
        "bert": bert_available(),
    }


# ---------------- HELPERS ---------------- #
def _decision_to_confidence(scores):
    best = float(np.max(scores))
    return round(100 / (1 + np.exp(-best)), 2)


# ---------------- CLANG ERROR EXTRACTION ---------------- #
def extract_error_from_c_file(file_path: str) -> str:
    lexical_markers = (
        "stray",
        "missing terminating",
        "unterminated",
        "invalid suffix",
        "invalid digit",
        "invalid character",
        "unknown escape",
        "character constant too long",
        "expected expression before",
    )
    try:
        result = subprocess.run(
            ["clang", "-fsyntax-only", "-fno-diagnostics-color", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )

        all_errors = []
        for line in result.stderr.splitlines():
            if "error:" in line.lower():
                msg = re.sub(r".*error:\s*", "", line, flags=re.IGNORECASE).strip()
                if msg:
                    all_errors.append(msg)

        # Prefer lexical-style diagnostics when available, otherwise fallback to first error.
        for msg in all_errors:
            lowered = msg.lower()
            if any(marker in lowered for marker in lexical_markers):
                return msg

        if all_errors:
            return all_errors[0]

        return ""

    except Exception as e:
        print("❌ Clang error:", e)
        return ""


# ---------------- AST EXTRACTION ---------------- #
def extract_ast_features(file_path: str, ast_columns):
    if not PYCPARSER_AVAILABLE:
        print("⚠️ pycparser not installed")
        return None

    try:
        ast = parse_file(
            file_path,
            use_cpp=True,
            cpp_path="gcc",
            cpp_args=[
                "-E",
                "-nostdinc",
                "-I", FAKE_LIBC_PATH   # 🔥 THIS IS THE FIX
            ]
        )

        visitor = ASTVisitor()
        visitor.visit(ast)

        counts = Counter(visitor.nodes)

        if ast_columns is None:
            return None

        ordered = [counts.get(col, 0) for col in ast_columns]
        return np.array(ordered).reshape(1, -1)

    except Exception as e:
        print("⚠️ AST extraction failed:", e)
        return None


# ---------------- MODEL 1 ---------------- #
def _predict_supervised(text: str):
    vec = _registry.load("supervised_vectorizer")
    model = _registry.load("supervised_model")

    X = vec.transform([text])
    label = model.predict(X)[0]
    scores = model.decision_function(X)[0]

    return {
        "label": label,
        "confidence": _decision_to_confidence(scores),
        "model_used": "supervised_model"
    }


# ---------------- MODEL 2 ---------------- #
def _predict_svm(text: str, file_path: str = None):
    vec = _registry.load("svm_vectorizer")
    model = _registry.load("svm_classifier")
    columns = _registry.load("ast_columns")

    tfidf = vec.transform([text])

    # AST extraction
    ast_features = None
    if file_path and columns is not None:
        ast_features = extract_ast_features(file_path, columns)

    # fallback if AST fails
    if ast_features is None:
        ast_dim = len(columns) if columns is not None else 42
        ast_features = np.zeros((1, ast_dim))

    X = sp.hstack([tfidf, sp.csr_matrix(ast_features)])

    label = model.predict(X)[0]
    scores = model.decision_function(X)[0]

    return {
        "label": label,
        "confidence": _decision_to_confidence(scores),
        "model_used": "svm_classifier (TF-IDF + AST)"
    }


# ---------------- MODEL 3 ---------------- #
def _predict_stacking(text: str):
    artifact = _registry.load("stacking_model")
    model = artifact
    vec = _registry.load("stacking_vectorizer")

    # Some saved artifacts store (model, vectorizer) as a tuple.
    if isinstance(artifact, tuple):
        if len(artifact) >= 1:
            model = artifact[0]
        if len(artifact) >= 2 and artifact[1] is not None:
            vec = artifact[1]

    if model is None or vec is None:
        raise RuntimeError("Stacking artifact missing model/vectorizer.")

    X = vec.transform([text])
    label = model.predict(X)[0]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = scores[0] if getattr(scores, "ndim", 1) > 1 else scores
        confidence = _decision_to_confidence(scores)
    elif hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        confidence = round(float(np.max(probs)) * 100, 2)
    else:
        confidence = 0.0

    return {
        "label": label,
        "confidence": confidence,
        "model_used": "stacking_model (TF-IDF)"
    }


# ---------------- CORE FUNCTION ---------------- #
def predict_both(text: str = None, file_content: str = None) -> List[Dict]:
    results = []

    print("\n🔍 Running multi-model prediction...")

    tmp_path = None
    effective_text = (text or "").strip()

    # ---------- FILE INPUT ----------
    if file_content:
        with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w") as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        if not effective_text:
            effective_text = (extract_error_from_c_file(tmp_path) or "").strip()
            print("📂 Extracted error:", effective_text)
        else:
            print("📂 Using provided error text; temp file kept for AST/SVM path.")

    text = effective_text

    if not text:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return []

    # ---------- MODEL 1 ----------
    if _registry.supervised_available():
        try:
            res = _predict_supervised(text)
            print("✅ Supervised:", res)
            results.append(res)
        except Exception as e:
            print("❌ Supervised failed:", e)

    # ---------- MODEL 2 ----------
    if _registry.svm_available():
        try:
            res = _predict_svm(text, file_path=tmp_path)
            print("✅ SVM:", res)
            results.append(res)
        except Exception as e:
            print("❌ SVM failed:", e)

    # ---------- MODEL 3 ----------
    if _registry.stacking_available():
        try:
            res = _predict_stacking(text)
            print("✅ Stacking:", res)
            results.append(res)
        except Exception as e:
            print("❌ Stacking failed:", e)

    # ---------- MODEL 4 (BERT from train_bert.py artifacts) ----------
    if bert_available():
        try:
            res = _predict_bert(text)
            print("✅ BERT:", res)
            results.append(res)
        except Exception as e:
            print("❌ BERT failed:", e)

    print("📊 Models used:", len(results))

    # cleanup
    if tmp_path and os.path.exists(tmp_path):
        os.unlink(tmp_path)

    return results


# ---------------- SINGLE OUTPUT (for agent) ---------------- #
def predict_single(text: str = None, file_content: str = None) -> Dict:
    results = predict_both(text=text, file_content=file_content)

    if not results:
        return {
            "label": "unknown",
            "confidence": 0,
            "model_used": "none",
            "routing_reason": "No models available"
        }

    best = max(results, key=lambda x: x["confidence"])

    return {
        "label": best["label"],
        "confidence": best["confidence"],
        "model_used": best["model_used"],
        "routing_reason": "Selected highest confidence model",
        "low_conf": best["confidence"] < 60
    }