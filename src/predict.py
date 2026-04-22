
import os, sys, re, joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from preprocess   import clean_error
from security_map import get_security_info, get_risk_emoji

def _find(filename):
    for p in [os.path.join(BASE_DIR, "..", "models", filename),
              os.path.join(BASE_DIR, "models", filename)]:
        if os.path.exists(p): return os.path.abspath(p)
    raise FileNotFoundError(f"'{filename}' not found. Run train_supervised.py first.")

model      = joblib.load(_find("supervised_model.pkl"))
vectorizer = joblib.load(_find("tfidf_vectorizer.pkl"))

PHASE = {
    "lexical" : "Phase 1 - Lexical Analysis",
    "syntax"  : "Phase 2 - Syntax Analysis",
    "semantic": "Phase 3 - Semantic Analysis",
}

FIX_RULES = {
    "lexical": [
        ("stray",               "Remove the stray character or fix the escape sequence."),
        ("missing terminating", "Close the string or character literal properly."),
        ("invalid character",   "Remove or replace the invalid character."),
        ("unterminated",        "Check for an unclosed string, char literal, or comment."),
        ("invalid suffix",      "Remove the invalid suffix from the numeric literal."),
        ("unknown escape",      "Use a valid escape sequence."),
    ],
    "syntax": [
        ("expected ';'",        "Add a semicolon ; at the end of the statement."),
        ("expected '}'",        "Add a closing brace }."),
        ("expected '{'",        "Add an opening brace {."),
        ("expected identifier", "Provide a valid variable or function name."),
        ("expected expression", "Check for a missing or incomplete expression."),
        ("missing semicolon",   "Add a semicolon after the statement."),
    ],
    "semantic": [
        ("undeclared",          "Declare the variable or function before using it."),
        ("incompatible types",  "Check type compatibility or add an explicit cast."),
        ("too few arguments",   "Pass all required arguments to the function."),
        ("too many arguments",  "Remove the extra arguments from the function call."),
        ("redeclaration",       "Variable already declared - rename or remove duplicate."),
        ("division by zero",    "Check your divisor - it evaluates to zero."),
        ("array subscript",     "Ensure array index is within bounds."),
    ],
}
FALLBACK = {
    "lexical" : "Check for invalid characters, bad tokens, or unclosed literals.",
    "syntax"  : "Check brackets, semicolons, and grammar structure.",
    "semantic": "Check variable declarations, types, and function signatures.",
}

def _get_fix(msg, label):
    ml = msg.lower()
    for p,t in FIX_RULES.get(label,[]):
        if p in ml: return t
    return FALLBACK.get(label, "Review your code carefully.")

def _get_proba(vec):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        return {str(c): round(float(p)*100,1) for c,p in zip(model.classes_, probs)}
    scores = model.decision_function(vec)
    if scores.ndim == 2: scores = scores[0]
    scores = scores.astype(float)
    e = np.exp(scores - scores.max()); sm = e/e.sum()
    return {str(c): round(float(p)*100,1) for c,p in zip(model.classes_, sm)}


def predict_single(raw_msg: str) -> dict:
    
    cleaned  = clean_error(raw_msg)
    vec      = vectorizer.transform([cleaned])
    label    = str(model.predict(vec)[0])
    probs    = _get_proba(vec)
    top_conf = float(max(probs.values()))
    security = get_security_info(label, raw_msg)
    return {
        "label"     : label,
        "phase"     : PHASE.get(label, ""),
        "confidence": round(top_conf, 1),
        "all_probs" : probs,
        "fix"       : _get_fix(raw_msg, label),
        "low_conf"  : top_conf < 60.0,
        "security"  : security,
    }

def predict_batch(messages: list) -> list:
    """Classify a list of error messages."""
    return [predict_single(m) for m in messages]

if __name__ == "__main__":
    EXAMPLES = [
        "error: expected ';' before '}' token",
        "error: stray '\\' in program",
        "error: undeclared identifier 'x'",
        "error: array subscript is above array bounds",
        "error: incompatible types when assigning to type int from type char",
        "error: division by zero",
        "error: implicit declaration of function 'gets'",
    ]
    print("  Compiler Error Classifier + Security Mapping")
    
    print("Commands: examples | exit\n")
    while True:
        try: text = input(">> ").strip()
        except (EOFError, KeyboardInterrupt): print("\nExiting..."); break
        if not text: continue
        if text.lower() in ("exit","quit"): print("Exiting..."); break
        if text.lower() == "examples":
            for i,e in enumerate(EXAMPLES,1): print(f"  {i}. {e}")
            print(); continue
        r   = predict_single(text)
        sec = r["security"]
        print(f"\n{'='*60}")
        print(f"  Input      : {text}")
        print(f"  Prediction : {r['label'].upper()}")
        print(f"  Phase      : {r['phase']}")
        print(f"  Confidence : {r['confidence']}%"
              + ("  [Low confidence]" if r["low_conf"] else ""))
        print(f"\n  Fix : {r['fix']}")
        print(f"\n  Security : {get_risk_emoji(sec['risk'])} {sec['risk']}")
        if sec.get("name"):        print(f"  Vuln     : {sec['name']}")
        if sec.get("cwe"):         print(f"  CWE      : {sec['cwe']}")
        if sec.get("description"): print(f"  Detail   : {sec['description']}")
        print(f"{'='*60}\n")