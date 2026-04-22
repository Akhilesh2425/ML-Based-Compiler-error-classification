import os, re, html, pickle, shutil, tempfile, subprocess, numpy as np, joblib, streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO

try:
    from security import get_security_info
except ImportError:
    try:
        from security_map import get_security_info
    except ImportError:
        def get_security_info(label, msg):
            return {"risk": "None", "cwe": "", "name": "Module Missing", "description": "Place security.py in folder.", "recommendation": ""}

st.set_page_config(page_title="CEC | ML Compiler Error Classification", layout="wide", page_icon="⚙️")

st.markdown("""
<style>
    /* ── Global background & text ── */
    .stApp { background-color: #f5f7fa; color: #1a1f2e; }

    /* ── Fade-in animation ── */
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .main-content { animation: fadeIn 0.4s ease-out; }

    /* ── Result cards ── */
    .res-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #d0d7e3;
        text-align: center;
        height: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .res-card:hover { border-color: #2563eb; transform: translateY(-3px); box-shadow: 0 6px 18px rgba(37,99,235,0.12); }
    .res-big { font-size: 22px; font-weight: 700; color: #2563eb; margin-bottom: 5px; }
    .res-small { color: #5a6478; font-size: 14px; }

    /* ── Primary buttons (Streamlit default overrides) ── */
    div.stButton > button:first-child {
        background-color: #2563eb;
        border: 1px solid #1d4ed8;
        color: #ffffff;
        border-radius: 8px;
        font-weight: 600;
    }
    div.stButton > button:first-child:hover {
        background-color: #1d4ed8;
        border-color: #1e40af;
        color: #ffffff;
    }

    button[kind="primary"],
    button[data-testid="baseButton-primary"],
    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {
        background-color: #2563eb !important;
        border: 1px solid #1d4ed8 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        border-color: #1e40af !important;
        color: #ffffff !important;
    }

    /* ── Secondary / inactive nav buttons ── */
    div.stButton > button[kind="secondary"] {
        background-color: #e8edf5;
        border: 1px solid #c5cfe0;
        color: #2c3a52;
        border-radius: 8px;
        font-weight: 500;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #dce4f0;
        border-color: #2563eb;
        color: #1a3a8f;
    }

    button[kind="secondary"],
    button[data-testid="baseButton-secondary"],
    div.stButton > button[kind="secondary"],
    div.stFormSubmitButton > button[kind="secondary"] {
        background-color: #e8edf5 !important;
        border: 1px solid #c5cfe0 !important;
        color: #1e293b !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    button[kind="secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover,
    div.stButton > button[kind="secondary"]:hover,
    div.stFormSubmitButton > button[kind="secondary"]:hover {
        background-color: #dce4f0 !important;
        border-color: #2563eb !important;
        color: #1a3a8f !important;
    }
    button:disabled,
    button[disabled] {
        background-color: #e5e7eb !important;
        border-color: #cbd5e1 !important;
        color: #64748b !important;
        opacity: 1 !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] h2 { color: #f1f5f9 !important; }
    section[data-testid="stSidebar"] p  { color: #94a3b8 !important; }

    /* Sidebar nav buttons — primary (active) */
    section[data-testid="stSidebar"] div.stButton > button:first-child[kind="primary"] {
        background-color: #2563eb !important;
        border: 1px solid #1d4ed8 !important;
        color: #ffffff !important;
    }
    /* Sidebar nav buttons — secondary (inactive) */
    section[data-testid="stSidebar"] div.stButton > button:first-child {
        background-color: #334155 !important;
        border: 1px solid #475569 !important;
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:first-child:hover {
        background-color: #3b4f6b !important;
        border-color: #60a5fa !important;
        color: #ffffff !important;
    }

    /* ── Sidebar expander arrow & caption ── */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        color: #cbd5e1 !important;
        background-color: #273549 !important;
        border-radius: 6px;
    }
    section[data-testid="stSidebar"] .streamlit-expanderContent {
        background-color: #1e293b !important;
        border-top: 1px solid #334155;
    }
    section[data-testid="stSidebar"] svg { fill: #94a3b8 !important; stroke: #94a3b8 !important; }

    /* ── Main area expanders ── */
    .streamlit-expanderHeader {
        background-color: #eef2f9 !important;
        border-radius: 8px;
        color: #1e293b !important;
        font-weight: 600;
    }
    .streamlit-expanderContent {
        background-color: #f9fafc !important;
    }

    /* ── Text inputs / text areas ── */
    .stTextArea textarea, .stTextInput input {
        background-color: #ffffff !important;
        border: 1px solid #c5cfe0 !important;
        color: #1a1f2e !important;
        border-radius: 8px !important;
    }

    /* ── File uploader / Browse files button ── */
    div[data-testid="stFileUploader"] section {
        background-color: #f8f9fc !important;
        border: 1px dashed #c5cfe0 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stFileUploader"] button,
    section[data-testid="stFileUploaderDropzone"] button {
        background-color: #2563eb !important;
        border: 1px solid #1d4ed8 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stFileUploader"] button:hover,
    section[data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #1d4ed8 !important;
        border-color: #1e40af !important;
        color: #ffffff !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 1px solid #c5cfe0 !important;
        color: #1a1f2e !important;
        border-radius: 8px !important;
    }

    /* ── Metrics ── */
    div[data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #d0d7e3;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }

    /* ── Dataframe ── */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* ── Dividers ── */
    hr { border-color: #d0d7e3 !important; }

    /* ── Caption / small text ── */
    .stCaption { color: #5a6478 !important; }

    /* ── Download button ── */
    div.stDownloadButton > button {
        background-color: #16a34a !important;
        border: 1px solid #15803d !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 600;
    }
    div.stDownloadButton > button:hover {
        background-color: #15803d !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_LABELS_IN_ORDER = [
    "Supervised Model",
    "Bert",
    "Semi Lr",
    "Semi Nb",
    "Semi Svc",
    "Compiler Error Classifier",
]

def find_file(name):
    for p in [os.path.join(BASE_DIR, "models", name), os.path.join(BASE_DIR, name), os.path.join(BASE_DIR, "..", "models", name)]:
        if os.path.exists(p): return p
    return None

def resolve_bert_bundle_path():
    paths = [
        os.path.join(BASE_DIR, "..", "output_models", "model.pkl"),
        os.path.join(BASE_DIR, "..", "output_models", "BERT_model.pkl"),
    ]
    for p in paths:
        if os.path.isfile(p):
            return p
    return None

def resolve_bert_model_dir(bundle):
    if isinstance(bundle, dict):
        md = bundle.get("model_dir")
        if md and os.path.isdir(md):
            return md
    fallback = os.path.join(BASE_DIR, "..", "bert_model")
    return fallback if os.path.isdir(fallback) else None

def bert_available():
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: F401
    except ImportError:
        return False

    b_path = resolve_bert_bundle_path()
    if not b_path:
        return False
    try:
        with open(b_path, "rb") as f:
            bundle = pickle.load(f)
    except Exception:
        return False
    return resolve_bert_model_dir(bundle) is not None

@st.cache_resource
def load_bert_runtime():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    b_path = resolve_bert_bundle_path()
    if not b_path:
        raise RuntimeError("BERT metadata missing in output_models/model.pkl (or BERT_model.pkl).")

    with open(b_path, "rb") as f:
        bundle = pickle.load(f)
    model_dir = resolve_bert_model_dir(bundle)
    if not model_dir:
        raise RuntimeError("BERT model directory not found. Expected bert_model/.")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer, bundle

@st.cache_resource
def load_resources(model_path, vectorizer_path):
    if model_path:
        mdl = joblib.load(model_path)
        if isinstance(mdl, dict):
            return mdl.get("model"), mdl.get("vectorizer")
        elif vectorizer_path:
            return mdl, joblib.load(vectorizer_path)
    return None, None

def discover_model_options():
    default_vectorizer = find_file("tfidf_vectorizer.pkl")
    model_candidates = {
        "Supervised Model": ["supervised_model.pkl"],
        "Bert": [os.path.join("output_models", "BERT_model.pkl"), os.path.join("output_models", "model.pkl")],
        "Semi Lr": ["semi_LR.pkl", "semi_lr.pkl", "super_LR.pkl"],
        "Semi Nb": ["semi_NB.pkl", "semi_nb.pkl", "super_NB.pkl"],
        "Semi Svc": ["semi_SVC.pkl", "semi_svc.pkl", "super_SVC.pkl"],
        "Compiler Error Classifier": ["compiler_error_classifier.pkl"],
    }

    options = []
    for label in MODEL_LABELS_IN_ORDER:
        path = None
        for candidate in model_candidates[label]:
            if os.path.isabs(candidate):
                candidate_path = candidate
            else:
                candidate_path = find_file(candidate)
                if not candidate_path:
                    candidate_path = os.path.join(BASE_DIR, "..", candidate)
            if candidate_path and os.path.exists(candidate_path):
                path = candidate_path
                break

        if not path:
            continue

        vectorizer_path = default_vectorizer
        if label == "Bert":
            vectorizer_path = None

        options.append({
            "key": f"{label}:{os.path.relpath(path, BASE_DIR)}",
            "label": label,
            "model_path": path,
            "vectorizer_path": vectorizer_path,
        })
    return options

MODEL_OPTIONS = discover_model_options()

def clean(msg):
    msg = str(msg).lower()
    msg = re.sub(r"[^a-z0-9\s']", " ", msg)
    return re.sub(r"\s+", " ", msg).strip()

def sanitize_c_source(raw):
    code = raw.replace("\r", "")
    code = re.sub(r"^\s*#.*$", "", code, flags=re.MULTILINE)
    return code

def parse_c_error(code):
    try:
        from pycparser import c_parser
    except Exception:
        return None

    parser = c_parser.CParser()
    try:
        parser.parse(code, filename="<user_code>")
        return None
    except Exception as e:
        msg = str(e)
        m = re.search(r":(?P<line>\d+):(?P<col>\d+):", msg)
        if m:
            line_no = int(m.group("line"))
            col_no = int(m.group("col"))
            clean_msg = re.sub(r".*?:\d+:\d+:\s*", "", msg).strip()
            return clean_msg or msg, line_no, col_no
        return msg, None, None

def extract_error_from_c_source(source_code):
    def run_compiler_and_collect(c_code):
        tmp_local = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w", encoding="utf-8") as tmp:
                tmp.write(c_code)
                tmp_local = tmp.name
            cmd_local = [
                compiler,
                "-x", "c",
                "-std=c11",
                "-fsyntax-only",
                "-fno-diagnostics-color",
                "-Werror=implicit-function-declaration",
                tmp_local,
            ]
            result_local = subprocess.run(
                cmd_local,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )
            stderr_local = result_local.stderr or ""
            found = []
            for line in stderr_local.splitlines():
                m = re.search(r":(?P<line>\d+):(?P<col>\d+):\s*error:\s*(?P<msg>.*)", line, re.IGNORECASE)
                if m:
                    found.append((m.group("msg").strip(), int(m.group("line")), int(m.group("col"))))
                elif "error:" in line.lower():
                    msg = re.sub(r".*error:\s*", "", line, flags=re.IGNORECASE).strip()
                    found.append((msg, None, None))
            return found, stderr_local
        except Exception as ex:
            return [], str(ex)
        finally:
            if tmp_local and os.path.exists(tmp_local):
                try:
                    os.unlink(tmp_local)
                except Exception:
                    pass

    compiler = shutil.which("clang") or shutil.which("gcc")
    if not compiler:
        parse_result = parse_c_error(sanitize_c_source(source_code))
        if parse_result:
            fallback_msg, fallback_line, fallback_col = parse_result
            return fallback_msg, fallback_line, fallback_col, "Compiler missing; using parser diagnostics."
        return None, None, None, "No compiler found. Install clang or gcc and add it to PATH."

    try:
        sanitized_source = sanitize_c_source(source_code)
        sanitized_errors, sanitized_stderr = run_compiler_and_collect(sanitized_source)
        for msg, line_no, col_no in sanitized_errors:
            if is_noise_compiler_error(msg):
                continue
            return msg, line_no, col_no, sanitized_stderr

        if sanitized_errors:
            parse_result = parse_c_error(sanitized_source)
            if parse_result:
                fallback_msg, fallback_line, fallback_col = parse_result
                return fallback_msg, fallback_line, fallback_col, sanitized_stderr
            msg, line_no, col_no = sanitized_errors[0]
            return msg, line_no, col_no, sanitized_stderr

        parse_result = parse_c_error(sanitized_source)
        if parse_result:
            fallback_msg, fallback_line, fallback_col = parse_result
            return fallback_msg, fallback_line, fallback_col, sanitized_stderr
        return None, None, None, sanitized_stderr
    except Exception as e:
        return None, None, None, str(e)

def is_noise_compiler_error(msg):
    if not msg:
        return False
    lowered = msg.lower()
    noise_markers = [
        "directive",
        "preprocessor",
        "file not found",
        "no such file or directory",
        "fatal error",
        ".h'",
        ".h\"",
    ]
    return any(marker in lowered for marker in noise_markers)

def _highlight_span_on_line(line, col_one_based):
    if not line:
        return 0, 0
    idx = max(0, min(col_one_based - 1, len(line) - 1))
    ch = line[idx]
    if ch.isalnum() or ch == "_":
        left = idx
        while left > 0 and (line[left - 1].isalnum() or line[left - 1] == "_"):
            left -= 1
        right = idx + 1
        while right < len(line) and (line[right].isalnum() or line[right] == "_"):
            right += 1
        return left, right
    return idx, idx + 1

def render_code_preview_dark(code, error_line=None, error_col=None, error_caption=None):
    raw_lines = code.split("\n")
    width = max(2, len(str(len(raw_lines) or 1)))
    body_rows = []
    for i, ln in enumerate(raw_lines, start=1):
        num = str(i).rjust(width)
        is_err_line = error_line is not None and i == error_line
        line_bg = "rgba(220, 38, 38, 0.08)" if is_err_line else "transparent"
        if is_err_line and error_col is not None and error_col >= 1:
            a, b = _highlight_span_on_line(ln, error_col)
            before = html.escape(ln[:a])
            mid = html.escape(ln[a:b]) if a < b else ""
            after = html.escape(ln[b:])
            hl = "<span class='err-char'>&nbsp;</span>" if a >= b else f"<span class='err-char'>{mid}</span>"
            inner = before + hl + after
        else:
            inner = html.escape(ln)
        body_rows.append(
            f"<div class='code-line' style='background:{line_bg};'>"
            f"<span class='ln'>{html.escape(num)}</span>"
            f"<span class='sep'>│</span>"
            f"<span class='src'>{inner}</span>"
            f"</div>"
        )

    cap = f"<div class='err-caption'>{html.escape(error_caption)}</div>" if error_caption else ""
    st.markdown(
        f"""
<style>
.code-wrap {{
  background: #f8f9fc;
  color: #1e293b;
  border-radius: 10px;
  border: 1px solid #d0d7e3;
  font-family: Consolas, "Courier New", monospace;
  font-size: 13px;
  line-height: 1.5;
  overflow: auto;
  max-height: min(70vh, 640px);
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}}
.code-inner {{
  min-width: 100%;
  display: inline-block;
  padding: 12px 14px 14px 14px;
  box-sizing: border-box;
}}
.code-line {{
  display: flex;
  flex-wrap: nowrap;
  white-space: pre;
}}
.ln {{
  flex: 0 0 auto;
  width: {width + 1}ch;
  text-align: right;
  color: #94a3b8;
  user-select: none;
}}
.sep {{
  flex: 0 0 auto;
  color: #cbd5e1;
  padding: 0 10px 0 8px;
  user-select: none;
}}
.src {{
  flex: 1 1 auto;
  white-space: pre;
  overflow-x: auto;
  color: #1e293b;
}}
.err-char {{
  background: #ef4444;
  color: #ffffff;
  border-radius: 2px;
  padding: 0 1px;
}}
.err-caption {{
  margin: 0 0 10px 0;
  padding: 10px 12px;
  background: rgba(220, 38, 38, 0.08);
  border: 1px solid rgba(220, 38, 38, 0.30);
  border-radius: 8px;
  color: #b91c1c;
  font-size: 13px;
  font-weight: 500;
}}
</style>
{cap}
<div class="code-wrap"><div class="code-inner">
{"".join(body_rows)}
</div></div>
""",
        unsafe_allow_html=True,
    )

FIX_MAP = {
    "lexical": "Check for invalid characters, unclosed quotes, or illegal escape sequences.",
    "syntax": "Check for missing semicolons ( ; ), mismatched braces { }, or grammar errors.",
    "semantic": "Check for undeclared variables, type mismatches, or scope issues."
}

def classify(msg, selected_label, active_model, active_vectorizer):
    if selected_label == "Bert":
        if not bert_available():
            return "Unknown", 0.0, {}, "BERT is not ready (check output_models/model.pkl, bert_model/, torch, transformers)."
        import torch
        bert_model, tokenizer, bundle = load_bert_runtime()
        inputs = tokenizer(clean(msg), return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = bert_model(**inputs).logits
            p = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        top_idx = int(np.argmax(p))
        id2label = bundle.get("id2label") if isinstance(bundle, dict) else {}
        label = id2label.get(top_idx, id2label.get(str(top_idx), str(top_idx)))
        probs = {str(i): float(v) for i, v in enumerate(p)}
        return label, float(np.max(p)) * 100, probs, FIX_MAP.get(str(label).lower(), "Review logic and variable declarations.")

    if not active_model or not active_vectorizer:
        return "Unknown", 0.0, {}, "Model not loaded."
        
    v = active_vectorizer.transform([clean(msg)])
    label = active_model.predict(v)[0]
    if hasattr(active_model, "predict_proba"):
        p = active_model.predict_proba(v)[0]
    else:
        s = active_model.decision_function(v)
        s = np.asarray(s)
        if s.ndim == 1:
            s = np.stack([-s, s], axis=1)
        s0 = s[0]
        e = np.exp(s0 - np.max(s0))
        p = e / np.sum(e)
    classes = getattr(active_model, "classes_", None)
    if classes is None:
        probs = {}
    else:
        probs = dict(zip(classes, p))
    return label, max(p) * 100, probs, FIX_MAP.get(label, "Review logic and variable declarations.")

if "page" not in st.session_state: st.session_state.page = "Single"
if "history" not in st.session_state: st.session_state.history = []
if "selected_model_key" not in st.session_state and MODEL_OPTIONS:
    st.session_state.selected_model_key = MODEL_OPTIONS[0]["key"]
if "cfile_selected_model_key" not in st.session_state and MODEL_OPTIONS:
    st.session_state.cfile_selected_model_key = MODEL_OPTIONS[0]["key"]

selected_option = next((opt for opt in MODEL_OPTIONS if opt["key"] == st.session_state.get("selected_model_key")), None)
if selected_option is None and MODEL_OPTIONS:
    selected_option = MODEL_OPTIONS[0]
    st.session_state.selected_model_key = selected_option["key"]

model, vectorizer = load_resources(
    selected_option["model_path"] if selected_option else None,
    selected_option["vectorizer_path"] if selected_option else None,
)
selected_label = selected_option["label"] if selected_option else ""
model_ready = bert_available() if selected_label == "Bert" else bool(model and vectorizer)

with st.sidebar:
    st.markdown("""
        <h2 style='text-align: center; margin-bottom: 0px; color: #f1f5f9 !important;'>⚙️ CEC System</h2>
        <p style='text-align: center; color: #94a3b8 !important; font-size: 14px;'>Roll: 24CSB0A25</p>
    """, unsafe_allow_html=True)
    st.caption(f"Model status: {'✅ Loaded' if model_ready else '❌ Not loaded'}")
    st.write("---")
    
    if st.button("🔍 Single Analysis", use_container_width=True, type="primary" if st.session_state.page == "Single" else "secondary"):
        st.session_state.page = "Single"
        st.rerun()
    if st.button("📦 Batch Processing", use_container_width=True, type="primary" if st.session_state.page == "Batch" else "secondary"):
        st.session_state.page = "Batch"
        st.rerun()
    if st.button("📊 Analytics", use_container_width=True, type="primary" if st.session_state.page == "Analytics" else "secondary"):
        st.session_state.page = "Analytics"
        st.rerun()
    if st.button("🧾 C File Analysis", use_container_width=True, type="primary" if st.session_state.page == "CFile" else "secondary"):
        st.session_state.page = "CFile"
        st.rerun()


st.markdown('<div class="main-content">', unsafe_allow_html=True)

if st.session_state.page == "CFile":
    st.title("🧾 C File Error Classification")
    st.caption("Upload a C file, detect compiler error location, highlight line/column, and classify using selected model.")

    if MODEL_OPTIONS:
        cfile_labels = [opt["label"] for opt in MODEL_OPTIONS]
        cfile_idx = next(
            (i for i, opt in enumerate(MODEL_OPTIONS) if opt["key"] == st.session_state.get("cfile_selected_model_key")),
            0,
        )
        prev_cfile_key = st.session_state.get("cfile_selected_model_key")
        cfile_choice = st.selectbox("Select model for C-file analysis", cfile_labels, index=cfile_idx, key="cfile_model_selectbox")
        st.session_state.cfile_selected_model_key = next(opt["key"] for opt in MODEL_OPTIONS if opt["label"] == cfile_choice)
        if st.session_state.cfile_selected_model_key != prev_cfile_key:
            st.rerun()
    else:
        st.info("No model artifacts available for C-file analysis.")

    cfile_option = next((opt for opt in MODEL_OPTIONS if opt["key"] == st.session_state.get("cfile_selected_model_key")), None)
    cfile_model, cfile_vectorizer = load_resources(
        cfile_option["model_path"] if cfile_option else None,
        cfile_option["vectorizer_path"] if cfile_option else None,
    )
    cfile_label = cfile_option["label"] if cfile_option else ""
    cfile_model_ready = bert_available() if cfile_label == "Bert" else bool(cfile_model and cfile_vectorizer)

    if not cfile_model_ready:
        if cfile_label == "Bert":
            st.warning("BERT not ready. Ensure `output_models/model.pkl` (or `BERT_model.pkl`), `bert_model/`, and `torch/transformers` are available.")
        else:
            st.warning("Selected model/vectorizer is not ready.")

    cfile_upload = st.file_uploader("Upload `.c` file", type=["c"], key="page_c_file_uploader")
    run_cfile = st.button("Analyze C File & Classify", type="primary", disabled=not bool(cfile_upload) or not cfile_model_ready)

    if run_cfile and cfile_upload:
        source = cfile_upload.getvalue().decode("utf-8", errors="replace")
        sanitized_source = sanitize_c_source(source)
        parse_result = parse_c_error(sanitized_source)
        comp_msg, comp_line, comp_col, raw_diag = extract_error_from_c_source(source)

        err_msg, err_line, err_col = None, None, None
        if parse_result:
            err_msg, err_line, err_col = parse_result
        elif comp_msg and not is_noise_compiler_error(comp_msg):
            err_msg, err_line, err_col = comp_msg, comp_line, comp_col

        if not err_msg:
            st.warning("No compiler error found in this C file.")
            if raw_diag:
                with st.expander("Compiler diagnostics", expanded=False):
                    st.code(raw_diag, language="text")
        else:
            with st.spinner("Classifying extracted compiler error..."):
                lbl, conf, probs, fix = classify(err_msg, cfile_label, cfile_model, cfile_vectorizer)
                sec_info = get_security_info(lbl, err_msg)
                risk_level = sec_info.get("risk", "None")

            st.info(f"Detected error: `{err_msg}`")
            caption = None
            if err_line is not None and err_col is not None:
                caption = f"Parse error near line {err_line}, column {err_col}"
            elif err_line is not None:
                caption = f"Parse error near line {err_line}"
            render_code_preview_dark(sanitized_source, error_line=err_line, error_col=err_col, error_caption=caption)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="res-card"><div class="res-small">PHASE CLASSIFICATION</div><div class="res-big">{lbl.upper()}</div><div class="res-small">Confidence: {conf:.1f}%</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="res-card"><div class="res-small">SUGGESTED FIX</div><div class="res-big">🔧 Solution</div><div class="res-small" style="color:#5a6478;">{fix}</div></div>', unsafe_allow_html=True)

            if risk_level in ["Critical", "High"]:
                st.error(f"Security risk detected: {risk_level}")
            elif risk_level == "Medium":
                st.warning(f"Security warning: {risk_level}")
            elif risk_level == "Low" and lbl != "lexical":
                st.info("Security notice: Low risk")

            if probs:
                with st.expander("Probability table", expanded=False):
                    df_p = pd.DataFrame({"Category": list(probs.keys()), "Probability": list(probs.values())}).sort_values("Probability", ascending=False)
                    st.dataframe(df_p, use_container_width=True)
            with st.expander("Compiler diagnostics", expanded=False):
                st.code(raw_diag or "", language="text")

            st.session_state.history.append({
                "Error": err_msg,
                "Label": lbl,
                "Conf": conf,
                "Security Risk": risk_level
            })
            if len(st.session_state.history) > 200:
                st.session_state.history = st.session_state.history[-200:]

elif st.session_state.page == "Single":
    st.title(" ⚙️  Compiler Error Classification")
    st.caption("ML-Powered Lexical, Syntax, and Semantic Classification with Defense-in-Depth Security Mapping.")
    if MODEL_OPTIONS:
        model_labels = [opt["label"] for opt in MODEL_OPTIONS]
        current_idx = next(
            (i for i, opt in enumerate(MODEL_OPTIONS) if opt["key"] == st.session_state.get("selected_model_key")),
            0,
        )
        previous_key = st.session_state.get("selected_model_key")
        chosen_label = st.selectbox("Select model for classification", model_labels, index=current_idx)
        st.session_state.selected_model_key = next(
            opt["key"] for opt in MODEL_OPTIONS if opt["label"] == chosen_label
        )
        if st.session_state.selected_model_key != previous_key:
            st.rerun()
    else:
        st.info("No `.pkl` model files were found in `src/models/`, `src/`, or project `models/`.")

    if not model_ready:
        if selected_label == "Bert":
            st.warning("BERT not ready. Ensure `output_models/model.pkl` (or `BERT_model.pkl`), `bert_model/`, and `torch/transformers` are available.")
        else:
            st.warning("Model/vectorizer not found. Place the `.pkl` files under `src/models/` (or project `models/`) and restart the app.")

    st.subheader("Quick Examples")
    EXAMPLES = [
        "error: array subscript is above array bounds",
        "error: incompatible types when assigning to type int from type char",
        "error: division by zero",
        "error: implicit declaration of function 'gets'",
        "error: stray '\\' in program",
        "error: expected ';' before '}' token"
    ]
    
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.input_text = ex
            st.rerun()

    msg = st.text_area(
        "Compiler Error Message",
        value=st.session_state.get("input_text", ""),
        height=150,
        placeholder="Paste a compiler error line here…",
    )

    a1, a2, a3 = st.columns([1, 1, 2])
    with a1:
        run_single = st.button("Analyze & Classify", type="primary", use_container_width=True, disabled=not model_ready)
    with a2:
        clear_single = st.button("Clear", use_container_width=True)
    with a3:
        show_details = st.toggle("Show details (security + probabilities)", value=True)

    if clear_single:
        st.session_state.input_text = ""
        st.rerun()

    if run_single:
        if msg.strip():
            with st.spinner("Classifying…"):
                label, conf, probs, fix = classify(msg, selected_label, model, vectorizer)
                sec_info = get_security_info(label, msg)
                risk_level = sec_info.get("risk", "None")
            
            st.session_state.history.append({
                "Error": msg, 
                "Label": label, 
                "Conf": conf,
                "Security Risk": risk_level
            })
            if len(st.session_state.history) > 200:
                st.session_state.history = st.session_state.history[-200:]

            st.markdown("---")
            
            if conf < 50:
                st.info(f"Low confidence ({conf:.1f}%). Consider pasting more context or the full compiler line.")
            
            if risk_level in ["Critical", "High"]:
                cwe_str = f"[{sec_info['cwe']}] " if sec_info.get("cwe") else ""
                st.error(f"🚨 **SECURITY RISK DETECTED: {cwe_str}{sec_info.get('name', '')}**\n\n**Severity:** {risk_level}\n\n**Details:** {sec_info.get('description', '')}\n\n**Secure Fix:** {sec_info.get('recommendation', '')}")
            elif risk_level == "Medium":
                cwe_str = f"[{sec_info['cwe']}] " if sec_info.get("cwe") else ""
                st.warning(f"🟠 **SECURITY WARNING: {cwe_str}{sec_info.get('name', '')}**\n\n**Severity:** {risk_level}\n\n**Details:** {sec_info.get('description', '')}\n\n**Secure Fix:** {sec_info.get('recommendation', '')}")
            elif risk_level == "Low" and label != "lexical":
                st.info(f"🟡 **SECURITY NOTICE: {sec_info.get('name', '')}**\n\n**Details:** {sec_info.get('description', '')}\n\n**Secure Fix:** {sec_info.get('recommendation', '')}")
          
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="res-card"><div class="res-small">PHASE CLASSIFICATION</div><div class="res-big">{label.upper()}</div><div class="res-small">Confidence: {conf:.1f}%</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="res-card"><div class="res-small">SUGGESTED FIX</div><div class="res-big">🔧 Solution</div><div class="res-small" style="color:#5a6478;">{fix}</div></div>', unsafe_allow_html=True)

            if show_details:
                st.markdown("### Probability Distribution")
                if probs:
                    df_p = pd.DataFrame({"Category": list(probs.keys()), "Probability": list(probs.values())}).sort_values("Probability", ascending=True)
                    fig = px.bar(df_p, x="Probability", y="Category", orientation='h', color="Probability", color_continuous_scale='Blues', template="plotly_white")
                    fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("Probability table"):
                        st.dataframe(df_p.sort_values("Probability", ascending=False), use_container_width=True)
                else:
                    st.info("Probability details are not available for this model.")

                with st.expander("Security mapping details"):
                    sec_df = pd.DataFrame([{
                        "Phase": label,
                        "Risk": sec_info.get("risk", "None"),
                        "CWE": sec_info.get("cwe", ""),
                        "Name": sec_info.get("name", ""),
                        "Description": sec_info.get("description", ""),
                        "Recommendation": sec_info.get("recommendation", ""),
                    }])
                    st.dataframe(sec_df, use_container_width=True)

            with st.expander("Session history (last 200)"):
                if st.session_state.history:
                    df_h = pd.DataFrame(st.session_state.history)
                    st.dataframe(df_h, use_container_width=True)
                    st.download_button(
                        "📥 Download session history (CSV)",
                        data=df_h.to_csv(index=False).encode("utf-8"),
                        file_name="cec_session_history.csv",
                        mime="text/csv",
                    )
        else:
            st.warning("Please enter an error message.")

elif st.session_state.page == "Batch":
    st.title("Batch Error Classification")
    st.write("Analyze multiple compiler logs simultaneously for phase classification and security risks.")
    
    uploaded_file = st.file_uploader("Upload an error log file (.txt, .log)", type=["txt", "log"])
    st.markdown("---")
    batch_input = st.text_area("Or paste multiple errors here (one per line)", height=200)
    
    input_lines = []
    if uploaded_file:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        input_lines = [line.strip() for line in stringio.readlines() if line.strip()]
    elif batch_input:
        input_lines = [l.strip() for l in batch_input.split("\n") if l.strip()]

    if st.button("Run Batch Analysis", type="primary"):
        if input_lines:
            results = []
            for l in input_lines:
                lbl, c, _, fix = classify(l, selected_label, model, vectorizer)
                sec_info = get_security_info(lbl, l)
                results.append({
                    "Error": l, 
                    "Phase": lbl, 
                    "Confidence (%)": round(c, 2), 
                    "Security Risk": sec_info.get("risk", "None"),
                    "CWE": sec_info.get("cwe", ""),
                    "Fix": fix
                })
            
            res_df = pd.DataFrame(results)
            st.success(f"Processed {len(input_lines)} compiler errors.")
            st.dataframe(res_df, use_container_width=True)
            
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results as CSV", data=csv, file_name="compiler_error_analysis.csv", mime="text/csv")
        else:
            st.info("Upload a file or paste text to begin.")

elif st.session_state.page == "Analytics":
    st.title("System Analytics")
    if st.session_state.history:
        df_h = pd.DataFrame(st.session_state.history)
        
        if "Security Risk" in df_h.columns:
            risks_caught = len(df_h[(df_h["Security Risk"] != "None") & (df_h["Security Risk"] != "Low")])
            if risks_caught > 0:
                st.metric(label="🛡️ Total Critical/Medium Security Risks Prevented", value=risks_caught)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Error Phase Frequency")
            st.plotly_chart(px.pie(df_h, names="Label", hole=0.4, template="plotly_white"), use_container_width=True)
        with c2:
            st.subheader("Model Confidence Trend")
            st.plotly_chart(px.line(df_h, y="Conf", template="plotly_white", labels={"Conf": "Confidence %", "index": "Analysis Count"}), use_container_width=True)
        
        if st.button("Clear Session History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No classification data available yet. Run an analysis to populate insights.")

st.markdown('</div>', unsafe_allow_html=True)
