import os
import sys
import subprocess
import re
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(BASE_DIR, "..", "models", "semi_supervised_model.pkl")
vector_file = os.path.join(BASE_DIR, "..", "models", "tfidf_vectorizer.pkl")

try:
    model = joblib.load(model_file)
    vectorizer = joblib.load(vector_file)
except FileNotFoundError:
    print("Models not found")
    sys.exit(1)

LABEL_DESC = {
    "lexical": "Phase 1 Lexical Analysis",
    "syntax": "Phase 2 Syntax Analysis",
    "semantic": "Phase 3 Semantic Analysis",
}

SECURITY_RULES = {
    "array subscript": {
        "cwe": "CWE-119",
        "name": "Buffer Bounds Error",
        "severity": "CRITICAL",
        "desc": "Memory corruption risk"
    },
    "uninitialized": {
        "cwe": "CWE-457",
        "name": "Uninitialized Variable",
        "severity": "HIGH",
        "desc": "Unpredictable behavior"
    },
    "division by zero": {
        "cwe": "CWE-369",
        "name": "Divide By Zero",
        "severity": "MEDIUM",
        "desc": "Crash risk"
    }
}

def clean_error(text):
    text = text.lower()
    text = re.sub(r"\\[0-9a-f]+", " ", text)
    text = re.sub(r"'[^']{1,10}'", " TOKEN ", text)
    text = re.sub(r"\b\d+\b", " NUM ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def analyze_security(msg, label):
    if label != "semantic":
        return None
    msg_lower = msg.lower()
    for pattern, risk_data in SECURITY_RULES.items():
        if pattern in msg_lower:
            return risk_data
    return None

def display_analysis(raw_error, label, conf, security_alert):
    print("ERROR:", raw_error)
    print("TYPE:", label, "CONF:", round(conf * 100, 2))
    print("DESC:", LABEL_DESC.get(label, "Unknown"))
    if security_alert:
        print("SECURITY:", security_alert["severity"], security_alert["cwe"])

def scan_c_file(file_path):
    process = subprocess.run(
        ["gcc", "-fsyntax-only", "-Wall", file_path],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True
    )

    compiler_output = process.stderr

    if not compiler_output:
        print("No errors")
        return

    error_lines = [line for line in compiler_output.split('\n') if 'error:' in line or 'warning:' in line]

    for raw_line in error_lines:
        match = re.search(r"(error:|warning:)\s*(.*)", raw_line)
        if match:
            clean_msg = match.group(0)
            cleaned = clean_error(clean_msg)
            vec = vectorizer.transform([cleaned])
            label = model.predict(vec)[0]
            conf = model.predict_proba(vec).max(axis=1)[0]
            security_alert = analyze_security(clean_msg, label)
            display_analysis(raw_line.strip(), label, conf, security_alert)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scan_file.py file.c")
    else:
        target_file = sys.argv[1]
        if not os.path.exists(target_file):
            print("File not found")
        else:
            scan_c_file(target_file)