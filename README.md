# ML-Based Compiler Error Classification & Risk-Aware Troubleshooting

An ML system that classifies C compiler errors by type and, for semantic
errors specifically, assesses risk severity using real-world vulnerability
data — combined with an agentic layer that picks the best-performing model
for each error at inference time.

**Author:** Gurjigalla Akhilesh (Roll 24CSB0A25)

## How It Works

1. **Error Classification** — Multiple models were independently trained and
   evaluated to classify raw compiler errors by type:
   - Naive Bayes
   - Semi-supervised Naive Bayes
   - Logistic Regression
   - Semi-supervised Logistic Regression
   - Fine-tuned DistilBERT

2. **Semantic Error Risk Assessment** — For errors classified as semantic,
   a rule-based layer maps the error to a High/Medium/Low risk level, using
   mappings derived from real-world CWE (Common Weakness Enumeration)
   severity data. This runs independent of which model made the
   classification, and currently applies only to semantic error types.

3. **Agentic Model Selection** — Given an error, the agent runs it through
   the available models and selects the one reporting the highest
   classification confidence/accuracy, returning that model's prediction
   along with the risk assessment (if applicable).

See `UML Diagram.png` for the full system architecture.

## Tech Stack

- **Language:** Python, C (sample test files)
- **ML:** PyTorch, HuggingFace Transformers (DistilBERT), Scikit-learn
- **Data:** Custom-labeled dataset of compiler errors (`bert_dataset.csv`)

## Results

Five models were trained and compared (Naive Bayes, semi-supervised Naive
Bayes, Logistic Regression, semi-supervised Logistic Regression, and
DistilBERT). The agentic layer selects between them at inference time based
on per-error confidence rather than a single fixed "best" model — so overall
system accuracy depends on the error distribution seen at runtime rather
than one static number.


## Setup & Usage

```bash
git clone https://github.com/Akhilesh2425/ML-Based-Compiler-error-classification.git
cd ML-Based-Compiler-error-classification
pip install -r Requirements.txt
```

**Run the main pipeline:**
```bash
python src/main.py
```

**Run the agentic model on a single error:**
```bash
python src/agentic_model.py --error "error: expected ';' before '}' token"
```

**Run the agentic model on a full build log:**
```bash
python src/agentic_model.py --log-file build.log --json
```


## Project Structure

```
├── c_samples/         # Sample C files with intentional errors, used for testing
├── data/               # Training data
├── models/            # Saved model artifacts (NB, semi-NB, LR, semi-LR, DistilBERT)
├── src/
│   ├── main.py           # Entry point
│   ├── parser.py         # Extracts errors from logs
│   ├── fixer.py           # Builds fix actions and reasoning
│   └── agentic_model.py   # Orchestrates the agent loop, model selection
├── tests/             # Unit tests
└── train_bert.py       # DistilBERT fine-tuning script
```

## Limitations & Future Work

- Semantic risk assessment currently only applies to semantic error types
- Formal accuracy benchmarking with a proper held-out test split is planned —
  current per-model accuracy figures are still being validated
- CWE-based risk mapping is rule-based rather than learned; a data-driven
  risk model is a possible future direction

## Author

Gurjigalla Akhilesh — [B.Tech CSE, NIT Warangal](https://github.com/Akhilesh2425)
