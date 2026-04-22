
import os
import sys
import re
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from preprocess import clean_error
model_file  = os.path.join(BASE_DIR, "..", "models", "supervised_model.pkl")
vector_file = os.path.join(BASE_DIR, "..", "models", "tfidf_vectorizer.pkl")

model      = joblib.load(model_file)
vectorizer = joblib.load(vector_file)

LABEL_DESC = {
    "lexical"  : "Phase 1 – Lexical Analysis   | Invalid chars / malformed tokens",
    "syntax"   : "Phase 2 – Syntax Analysis    | Grammar errors (brackets, semicolons)",
    "semantic" : "Phase 3 – Semantic Analysis  | Type mismatch / undeclared variables",
}

FIX_RULES = {
    "lexical": [
        ("stray",               "Remove the stray character or fix the escape sequence."),
        ("missing terminating", "Close the string or character literal properly."),
        ("invalid character",   "Remove or replace the invalid character."),
        ("unterminated",        "Check for an unclosed string, char literal, or comment."),
        ("invalid suffix",      "Remove the invalid suffix from the numeric literal."),
        ("invalid digit",       "Use only valid digits for the base (e.g. 0-7 for octal)."),
        ("unknown escape",      "Use a valid escape sequence (\\n, \\t, \\\\, etc.)."),
    ],
    "syntax": [
        ("expected ';'",        "Add a semicolon ';' at the end of the statement."),
        ("expected '}'",        "Add a closing brace '}'."),
        ("expected '{'",        "Add an opening brace '{'."),
        ("expected '('",        "Add an opening parenthesis '('."),
        ("expected ')'",        "Add a closing parenthesis ')'."),
        ("expected identifier", "Provide a valid variable or function name."),
        ("expected expression", "Check for a missing or incomplete expression."),
        ("expected declaration","Add the required declaration."),
        ("missing semicolon",   "Add a semicolon ';' after the statement."),
    ],
    "semantic": [
        ("undeclared",          "Declare the variable or function before using it."),
        ("incompatible types",  "Check type compatibility or add an explicit cast."),
        ("too few arguments",   "Pass all required arguments to the function."),
        ("too many arguments",  "Remove the extra arguments from the function call."),
        ("redeclaration",       "Variable already declared — rename or remove duplicate."),
        ("conflicting types",   "Ensure function declaration and definition types match."),
        ("assignment makes",    "Add an explicit cast or fix the type mismatch."),
        ("division by zero",    "Check your divisor — it evaluates to zero."),
        ("array subscript",     "Ensure array index is within bounds and is an integer."),
    ],
}

FALLBACK = {
    "lexical"  : "Check for invalid characters, bad tokens, or unclosed literals.",
    "syntax"   : "Check brackets, semicolons, and overall grammar structure.",
    "semantic" : "Check variable declarations, types, and function signatures.",
}

EXAMPLES = [
    "error: expected ';' before '}' token",
    "error: undeclared identifier 'x'",
    "error: stray '\\' in program",
    "error: incompatible types when assigning to type 'int'",
    "error: missing terminating \" character",
    "error: too few arguments to function 'printf'",
    "error: invalid suffix abc on integer constant",
    "error: division by zero",
]


def suggest_fix(msg: str, label: str) -> str:
    msg_lower = msg.lower()
    for pattern, tip in FIX_RULES.get(label, []):
        if pattern in msg_lower:
            return tip
    return FALLBACK.get(label, "Review your code carefully.")




def render_bar(pct: float, width: int = 20) -> str:
    filled = round(pct / 100 * width)
    return "█" * filled


def show_result(raw_msg: str, label: str, probs: dict, fix: str):
    print("\n" + "=" * 60)
    print(f"  Input      : {raw_msg}")
    print(f"  Prediction : {label.upper()}")
    print(f"  Compiler   : {LABEL_DESC.get(label, 'N/A')}")

    top_conf = max(probs.values())
    if top_conf < 0.60:
        print(f"\n   Low confidence ({top_conf*100:.1f}%) — prediction may be uncertain")

    print("\n  Confidence Breakdown:")
    for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
        pct = prob * 100
        bar = render_bar(pct)
        marker = " ◀" if cls == label else ""
        print(f"    {cls:<10}  {bar}  {pct:5.1f}%{marker}")

    print(f"\n  Suggested Fix:")
    print(f"    {fix}")
    print("=" * 60 + "\n")

print("=" * 60)
print("  Compiler Error Classifier  –  CLI Demo")
print("  Roll: 24CSB0A25")
print("=" * 60)
print("\n  Commands: examples | help | quit\n")


while True:

    try:
        text = input(">> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        break

    if not text:
        continue

    cmd = text.lower()

    if cmd == "quit" or cmd == "exit":
        print("Exiting...")
        break

    if cmd == "help":
        print("\nAvailable commands:")
        print("  examples  – show sample error messages")
        print("  quit      – exit the program")
        print("  <error>   – type any compiler error message to classify it\n")
        continue

    if cmd == "examples":
        print("\nSample compiler errors:")
        for i, e in enumerate(EXAMPLES, 1):
            print(f"  {i}. {e}")
        print()
        continue

    cleaned = clean_error(text)

    vec   = vectorizer.transform([cleaned])
    label = model.predict(vec)[0]

    if hasattr(model, "predict_proba"):
        prob_arr = model.predict_proba(vec)[0]
        probs    = dict(zip(model.classes_, prob_arr))
    else:
        decision = model.decision_function(vec)[0]
        exp_d    = np.exp(decision - decision.max())
        softmax  = exp_d / exp_d.sum()
        probs    = dict(zip(model.classes_, softmax))

    fix = suggest_fix(text, label)
    show_result(text, label, probs, fix)