
from __future__ import annotations

import html
import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from agentic import ErrorClassificationAgent
from complexity_analyzer import analyze_empirical_complexity
from green_metrics import GreenResult, GreenTracker, render_green_dashboard_html
from predict import get_model_status


def sanitize_c_source(raw: str) -> str:
    """Normalize newlines and blank preprocessor lines without shifting line numbers."""
    code = raw.replace("\r", "")
    code = re.sub(r"^\s*#.*$", "", code, flags=re.MULTILINE)
    return code


MAX_MERMAID_NODES = 220


def _pycparser_available() -> bool:
    try:
        from pycparser import c_parser  # noqa: F401

        return True
    except Exception:
        return False


def _parse_c_ast(code: str) -> Tuple[Any, Optional[Tuple[int, int, str]]]:
    """
    Parse sanitized C with pycparser.
    Returns (ast_or_none, error_tuple) where error is (line, col, message).
    """
    try:
        from pycparser import c_parser  # type: ignore
    except Exception:
        return None, None

    parser = c_parser.CParser()
    try:
        ast = parser.parse(code, filename="<user_code>")
        return ast, None
    except Exception as e:
        msg = str(e)
        m = re.search(r":(?P<line>\d+):(?P<col>\d+):", msg)
        if m:
            return None, (int(m.group("line")), int(m.group("col")), msg)
        return None, (1, 1, msg)


def ast_to_mermaid_flowchart(ast: Any) -> str:
    """Recursively walk pycparser AST and build a Mermaid graph TD string."""

    lines: List[str] = ["graph TD"]
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"n{counter}"

    def mermaid_escape_label(s: str) -> str:
        s = s.replace('"', "'")
        s = re.sub(r"[\[\]\(\)\{\}]", " ", s)
        return s[:80]

    def walk(node: Any, parent_mid: Optional[str] = None) -> None:
        if counter >= MAX_MERMAID_NODES:
            return
        if node is None:
            return
        mid = next_id()
        label = mermaid_escape_label(node.__class__.__name__)
        lines.append(f'  {mid}["{label}"]')
        if parent_mid is not None:
            lines.append(f"  {parent_mid} --> {mid}")
        if counter >= MAX_MERMAID_NODES:
            return
        try:
            children = list(node.children())
        except Exception:
            return
        for _, child in children:
            walk(child, mid)

    walk(ast)
    if counter >= MAX_MERMAID_NODES:
        lines.append('  note["… truncated for performance …"]')
    return "\n".join(lines)


def render_mermaid(chart: str) -> None:
    """Render Mermaid via streamlit-mermaid when installed, else CDN + components."""
    if not chart.strip():
        st.info("No AST diagram to render.")
        return
    try:
        from streamlit_mermaid import st_mermaid  # type: ignore

        st_mermaid(chart, height=520)
        return
    except Exception:
        pass

    payload = json.dumps(chart)
    html_doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>body{{margin:0;background:#ffffff;}} #m{{padding:12px;color:#1e293b;}}</style>
</head><body>
<div id="m"></div>
<script>
(function() {{
  const spec = {payload};
  const el = document.getElementById("m");
  el.className = "mermaid";
  el.textContent = spec;
  mermaid.initialize({{ startOnLoad: false, theme: "default", securityLevel: "loose" }});
  mermaid.run({{ nodes: [el] }}).catch(function(e) {{
    el.textContent = "Mermaid render failed: " + (e && e.message ? e.message : String(e));
  }});
}})();
</script>
</body></html>"""
    components.html(html_doc, height=540, scrolling=True)


def _highlight_span_on_line(line: str, col_one_based: int) -> Tuple[int, int]:
    """Return (start, end) slice for highlight: word if alphanumeric, else single char."""
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


def render_code_preview_dark(
    code: str,
    error_line: Optional[int] = None,
    error_col: Optional[int] = None,
    error_caption: Optional[str] = None,
) -> None:
    """
    Responsive light code block with line numbers, optional exact error highlight.
    """
    raw_lines = code.split("\n")
    width = max(2, len(str(len(raw_lines) or 1)))

    body_rows: List[str] = []
    for i, ln in enumerate(raw_lines, start=1):
        num = str(i).rjust(width)
        is_err_line = error_line is not None and i == error_line
        line_bg = "rgba(220, 38, 38, 0.08)" if is_err_line else "transparent"

        if is_err_line and error_col is not None and error_col >= 1:
            a, b = _highlight_span_on_line(ln, error_col)
            before = html.escape(ln[:a])
            mid = html.escape(ln[a:b]) if a < b else ""
            after = html.escape(ln[b:])
            if a >= b:
                hl = "<span class='err-char'>&nbsp;</span>"
            else:
                hl = f"<span class='err-char'>{mid}</span>"
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

    cap = ""
    if error_caption:
        cap = f"<div class='err-caption'>{html.escape(error_caption)}</div>"

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


def analyze_source_block(raw: str) -> str:
    """
    Returns (sanitized_code, ast_or_none, parse_error, mermaid_or_none).
    If the raw translation unit fails to parse but pycparser is installed, we try a
    one-off wrapper function so statement-level snippets still produce a Mermaid tree.
    """
    sanitized = sanitize_c_source(raw)
    ast, err = _parse_c_ast(sanitized)
    mermaid_str: Optional[str] = None
    if ast is not None:
        try:
            mermaid_str = ast_to_mermaid_flowchart(ast)
        except Exception as ex:
            safe = re.sub(r"[^\w\s\-.,:+]", " ", str(ex))[:100].strip()
            mermaid_str = f'graph TD\n  x["AST to Mermaid failed {safe}"]'
    elif _pycparser_available() and err is not None and sanitized.strip():
        wrapped = "void __streamlit_ast_wrap__(void) {\n" + sanitized + "\n}\n"
        ast_wrap, _ = _parse_c_ast(wrapped)
        if ast_wrap is not None:
            try:
                mermaid_str = ast_to_mermaid_flowchart(ast_wrap)
            except Exception:
                mermaid_str = None
    return sanitized, ast, err, mermaid_str


def extract_multiple_errors(raw_error_text: str) -> List[str]:
    """
    Extract individual error messages from multiline compiler logs.
    Deduplicates while preserving order.
    """
    lines = [ln.strip() for ln in raw_error_text.splitlines() if ln.strip()]
    extracted: List[str] = []

    for ln in lines:
        if re.search(r"\berror\b", ln, flags=re.IGNORECASE):
            m = re.search(r"\berror\s*:\s*(.+)$", ln, flags=re.IGNORECASE)
            extracted.append((m.group(1) if m else ln).strip())
        elif ln.startswith("- ") or ln.startswith("* "):
            extracted.append(ln[2:].strip())
        elif "undefined reference" in ln.lower():
            extracted.append(ln.strip())

    if not extracted and raw_error_text.strip():
        parts = [
            p.strip()
            for p in re.split(r"\n{2,}|;\s+(?=[a-zA-Z])", raw_error_text)
            if p.strip()
        ]
        extracted = parts[:]

    deduped: List[str] = []
    seen = set()
    for msg in extracted:
        key = msg.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(msg)
    return deduped


def render_multi_error_results(
    rows: List[Dict[str, Any]],
    green_result: Optional[GreenResult] = None,
    complexity: Optional[dict] = None,
) -> None:
    st.success(f"Multi-error analysis complete ({len(rows)} errors)")

    labels = [str(r.get("final_label", "unknown")) for r in rows]
    counts = Counter(labels)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total errors", str(len(rows)))
    c2.metric("Top class", counts.most_common(1)[0][0] if counts else "—")
    c3.metric("Unique classes", str(len(counts)))

    st.markdown("**Predictions by error**")
    view_rows = [
        {
            "Error #": r["error_index"],
            "Detected message": r["error_text"],
            "Predicted class": r["final_label"],
            "Confidence %": r["confidence"],
            "Model used": r["model_used"],
        }
        for r in rows
    ]
    st.dataframe(pd.DataFrame(view_rows), use_container_width=True, hide_index=True)

    with st.expander("Advice per detected error", expanded=False):
        for r in rows:
            st.markdown(f"**Error {r['error_index']} — {r['final_label']} ({r['confidence']}%)**")
            st.caption(r["error_text"])
            st.write(r["advice"])
            st.markdown("---")

    st.markdown("### Green AI Metrics")
    green_history = st.session_state.get("green_history", [])
    tracker = st.session_state.get("green_tracker")
    methods = tracker.available_methods if tracker else ["estimated"]
    dash = render_green_dashboard_html(
        current=green_result,
        history=green_history,
        pinpoint_records=tracker.pinpoint_report() if tracker else [],
        available_methods=methods,
    )
    components.html(dash, height=420, scrolling=False)

    render_complexity_section(complexity)


def render_complexity_section(complexity: Optional[dict]) -> None:
    st.markdown("### Time Complexity (Empirical)")
    if not complexity:
        st.info("Run an analysis to estimate time complexity.")
        return

    if not complexity.get("computable", False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated Big-O", "N/A")
        c2.metric("Runs used", "0")
        c3.metric("Average runtime", "N/A")
        st.caption("Complexity is shown only for valid parsable C code.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated Big-O", str(complexity.get("estimated_big_o", "N/A")))
    c2.metric("Runs used", str(complexity.get("runs_used", 0)))
    c3.metric("Average runtime", f"{complexity.get('average_runtime_ms', 0.0):.3f} ms")
    st.caption(str(complexity.get("warning", "Includes empirical estimation; may vary based on environment.")))
    st.caption(str(complexity.get("message", "")))

    points = complexity.get("points") or []
    if points:
        df = pd.DataFrame(points, columns=["Input Size", "Runtime (ms)"])
        st.line_chart(df.set_index("Input Size"), height=180)


def render_analysis_result(
    result: dict,
    green_result: Optional[GreenResult] = None,
    complexity: Optional[dict] = None,
) -> None:
    st.success("Analysis complete")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Final label", str(result.get("final_label", "—")))
    with c2:
        st.metric("Confidence", f"{result.get('confidence', 0)}%")
    with c3:
        st.metric("Model used", str(result.get("model_used", "—")))

    st.markdown("**Agent decision**")
    st.caption(str(result.get("agent_decision", "")))

    st.markdown("##### Suggested fix")
    st.markdown(
        f'<div style="background:#ffffff;border:1px solid #d0d7e3;border-radius:10px;padding:14px 16px;color:#1e293b;box-shadow:0 2px 8px rgba(0,0,0,0.06);">{html.escape(str(result.get("advice", "")))}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Model comparison", expanded=False):
        outs = result.get("all_model_outputs") or []
        if not outs:
            st.warning("No model returned a prediction (check sidebar).")
        by_key = {}
        for model in outs:
            name = str(model.get("model_used", ""))
            lowered = name.lower()
            if "supervised_model" in lowered:
                by_key["supervised"] = model
            elif "svm_classifier" in lowered:
                by_key["svm"] = model
            elif "stacking_model" in lowered:
                by_key["stacking"] = model
            elif "bert" in lowered:
                by_key["bert"] = model
            else:
                by_key[name] = model

        ordered_models = [
            ("Supervised (TF-IDF)", "supervised"),
            ("Linear SVM (TF-IDF + AST)", "svm"),
            ("Stacking (stacking_model.pkl)", "stacking"),
            ("Fine-tuned BERT", "bert"),
        ]

        for title, key in ordered_models:
            m = by_key.get(key)
            if m:
                st.markdown(f"**{title}**")
                st.write(f"Prediction: **{m.get('label')}** — {m.get('confidence')}%")
            else:
                st.markdown(f"**{title}**")
                st.caption("Not used / unavailable")

    if result.get("low_confidence"):
        st.warning("Low confidence prediction — consider manual review.")

    st.markdown("### Green AI Metrics")
    green_history = st.session_state.get("green_history", [])
    tracker = st.session_state.get("green_tracker")
    methods = tracker.available_methods if tracker else ["estimated"]
    dash = render_green_dashboard_html(
        current=green_result,
        history=green_history,
        pinpoint_records=tracker.pinpoint_report() if tracker else [],
        available_methods=methods,
    )
    components.html(dash, height=420, scrolling=False)

    render_complexity_section(complexity)


st.set_page_config(
    page_title="Agentic Compiler Error Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  .stApp {
    background-color: #ffffff;
    color: #1a1f2e;
  }

  .block-container {
    padding-top: 1.2rem;
  }

  h1, h2, h3, h4, h5, h6, p, label, span {
    color: #1a1f2e;
  }

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

  div.stButton > button:first-child:focus {
    box-shadow: 0 0 0 0.2rem rgba(37, 99, 235, 0.25);
    color: #ffffff;
  }

  div.stButton > button:first-child[kind="secondary"] {
    background-color: #e8edf5;
    border: 1px solid #c5cfe0;
    color: #2c3a52;
  }

  div.stButton > button:first-child[kind="secondary"]:hover {
    background-color: #dce4f0;
    border-color: #2563eb;
    color: #1a3a8f;
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

  .stTextArea textarea,
  .stTextInput input,
  div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border-color: #c5cfe0 !important;
    color: #1a1f2e !important;
    border-radius: 8px !important;
  }

  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
  }

  .stTabs [data-baseweb="tab"] {
    background-color: #e8edf5;
    border: 1px solid #c5cfe0;
    border-radius: 8px 8px 0 0;
    color: #2c3a52;
    font-weight: 600;
  }

  .stTabs [aria-selected="true"] {
    background-color: #ffffff;
    color: #2563eb;
    border-bottom-color: #ffffff;
  }

  section[data-testid="stSidebar"] {
    background-color: #1e293b !important;
    border-right: 1px solid #334155;
  }

  section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
  }

  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] .stCaption {
    color: #94a3b8 !important;
  }

  .streamlit-expanderHeader {
    background-color: #eef2f9 !important;
    border-radius: 8px;
    color: #1e293b !important;
    font-weight: 600;
  }

  .streamlit-expanderContent {
    background-color: #f9fafc !important;
  }

  div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #d0d7e3;
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
  }

  div[data-testid="stFileUploader"] section {
    background-color: #f8f9fc;
    border: 1px dashed #c5cfe0;
    border-radius: 10px;
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

  div.stDownloadButton > button {
    background-color: #16a34a !important;
    border: 1px solid #15803d !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
  }

  div.stDownloadButton > button:hover {
    background-color: #15803d !important;
    border-color: #166534 !important;
    color: #ffffff !important;
  }

  .stDataFrame {
    border-radius: 8px;
    overflow: hidden;
  }

  hr {
    border-color: #d0d7e3 !important;
  }
</style>
""",
    unsafe_allow_html=True,
)

agent = ErrorClassificationAgent()
_status = get_model_status()
if "green_tracker" not in st.session_state:
    st.session_state["green_tracker"] = GreenTracker()
if "green_history" not in st.session_state:
    st.session_state["green_history"] = []

with st.sidebar:
    st.markdown("### Model status")
    st.markdown("---")
    sup = _status.get("supervised", False)
    svm = _status.get("svm", False)
    stacking = _status.get("stacking", False)
    bert = _status.get("bert", False)
    st.markdown(
        f"{'🟢' if sup else '🔴'} **Supervised (TF-IDF)** — {'ready' if sup else 'artifacts missing'}"
    )
    st.markdown(
        f"{'🟢' if svm else '🔴'} **Linear SVM (TF-IDF + AST)** — {'ready' if svm else 'artifacts missing'}"
    )
    st.markdown(
        f"{'🟢' if stacking else '🔴'} **Stacking Model (TF-IDF)** — {'ready' if stacking else 'artifacts missing'}"
    )
    st.markdown(
        f"{'🟢' if bert else '🔴'} **Fine-tuned BERT** — {'ready' if bert else 'train or install torch/transformers'}"
    )
    st.caption("BERT expects `output_models/model.pkl` and the `bert_model/` folder at project root.")
    tracker = st.session_state["green_tracker"]
    st.caption(f"Green tracking method: `{tracker.method}`")

st.title("Agentic Compiler Error Classifier")
st.caption(
    "Sanitized C parsing (pycparser) · dark code preview · Mermaid AST · multi-model agent"
)

tab_file, tab_paste = st.tabs(["Upload .c file", "Paste error / code"])

with tab_file:
    up = st.file_uploader("Drop a `.c` file", type=["c"])
    if up:
        raw_file = up.getvalue().decode("utf-8", errors="replace")
        sanitized, _ast, parse_err, mermaid_str = analyze_source_block(raw_file)

        st.subheader("Preview (sanitized for parser — line numbers match diagnostics)")
        if parse_err:
            el, ec, emsg = parse_err
            render_code_preview_dark(
                sanitized,
                error_line=el,
                error_col=ec,
                error_caption=f"Parse error near line {el}, column {ec}",
            )
            with st.expander("Parse message", expanded=False):
                st.code(emsg or "", language="text")
        else:
            render_code_preview_dark(sanitized)

        with st.expander("Abstract syntax tree (Mermaid)", expanded=True):
            if mermaid_str:
                render_mermaid(mermaid_str)
            elif not _pycparser_available():
                st.info("Install `pycparser` in the same environment you use for `streamlit run` (e.g. `pip install pycparser`).")
            elif parse_err:
                st.warning(
                    "pycparser could not build an AST from this file. Expand **Parse message** for details, "
                    "or fix the syntax issue above."
                )
            else:
                st.info("No C source to diagram yet.")

        if st.button("🚀 Analyze", key="btn_file", type="primary"):
            with st.spinner("Running agentic classification…"):
                parse_text = parse_err[2] if parse_err else None
                payload = (parse_text or "") + "\n" + (sanitized or "")
                tracker = st.session_state["green_tracker"]
                tracker.start("file_analysis")
                with tracker.pinpoint("agent.classify"):
                    result = agent.classify(text=parse_text, file_content=sanitized)
                green_result = tracker.stop()
                st.session_state["green_history"].append(green_result)
                complexity = analyze_empirical_complexity(
                    code=sanitized,
                    ast_root=_ast,
                    parse_error=parse_err,
                ).to_dict()
            render_analysis_result(result, green_result=green_result, complexity=complexity)

with tab_paste:
    err_text = st.text_area("Compiler / toolchain error message", height=140, placeholder="error: …")
    code_text = st.text_area("Optional: related C source", height=260, placeholder="#include …\nint main() { …")
    multi_error_mode = st.checkbox(
        "Enable multi-error detection",
        value=False,
        help="Split compiler logs into multiple errors and classify each one.",
    )

    if code_text.strip():
        sanitized_p, _astp, parse_errp, mermaid_p = analyze_source_block(code_text)
        st.subheader("Code preview")
        if parse_errp:
            el, ec, emsg = parse_errp
            render_code_preview_dark(
                sanitized_p,
                error_line=el,
                error_col=ec,
                error_caption=f"Parse error near line {el}, column {ec}",
            )
            with st.expander("Parse message", expanded=False):
                st.code(emsg or "", language="text")
        else:
            render_code_preview_dark(sanitized_p)

        with st.expander("Abstract syntax tree (Mermaid)", expanded=False):
            if mermaid_p:
                render_mermaid(mermaid_p)
            elif not _pycparser_available():
                st.info("Install `pycparser` in the same environment you use for `streamlit run` (e.g. `pip install pycparser`).")
            elif parse_errp:
                st.warning(
                    "pycparser could not build an AST from this snippet. Expand **Parse message** for details, "
                    "or fix the syntax issue above."
                )
            else:
                st.info("No C source to diagram yet.")
    else:
        sanitized_p = ""

    if st.button("🚀 Analyze", key="btn_paste", type="primary"):
        if not err_text.strip() and not code_text.strip():
            st.warning("Paste an error message and/or C code, then run Analyze.")
        else:
            with st.spinner("Running agentic classification…"):
                fc = sanitized_p if sanitized_p.strip() else None
                parse_text = parse_errp[2] if code_text.strip() and parse_errp else None
                manual_text = err_text if err_text.strip() else None
                if multi_error_mode and manual_text:
                    error_items = extract_multiple_errors(manual_text)
                    if not error_items:
                        error_items = [manual_text]

                    payload = manual_text + "\n" + (fc or "")
                    tracker = st.session_state["green_tracker"]
                    tracker.start("paste_multi_error_analysis")
                    rows: List[Dict[str, Any]] = []
                    with tracker.pinpoint("agent.classify.multi"):
                        for i, msg in enumerate(error_items, start=1):
                            r = agent.classify(text=msg, file_content=fc)
                            rows.append(
                                {
                                    "error_index": i,
                                    "error_text": msg,
                                    "final_label": r.get("final_label", "unknown"),
                                    "confidence": r.get("confidence", 0),
                                    "model_used": r.get("model_used", "none"),
                                    "advice": r.get("advice", ""),
                                }
                            )
                    green_result = tracker.stop()
                    st.session_state["green_history"].append(green_result)
                    complexity = analyze_empirical_complexity(
                        code=fc or "",
                        ast_root=_astp if code_text.strip() else None,
                        parse_error=parse_errp if code_text.strip() else (1, 1, "missing code"),
                    ).to_dict()
                    render_multi_error_results(rows, green_result=green_result, complexity=complexity)
                else:
                    payload = (manual_text or parse_text or "") + "\n" + (fc or "")
                    tracker = st.session_state["green_tracker"]
                    tracker.start("paste_analysis")
                    with tracker.pinpoint("agent.classify"):
                        result = agent.classify(text=manual_text or parse_text, file_content=fc)
                    green_result = tracker.stop()
                    st.session_state["green_history"].append(green_result)
                    complexity = analyze_empirical_complexity(
                        code=fc or "",
                        ast_root=_astp if code_text.strip() else None,
                        parse_error=parse_errp if code_text.strip() else (1, 1, "missing code"),
                    ).to_dict()
                    render_analysis_result(result, green_result=green_result, complexity=complexity)
