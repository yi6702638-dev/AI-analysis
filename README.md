# AI-based CV–Job Description Matching Analysis (Using NLP and Deep Learning Models) (Streamlit App)

A lightweight Streamlit web app that compares a **CV/Resume** against a **Job Description (JD)** and generates:

- **Match score (Model)** from trained deep-learning matcher  
- **TF-IDF Cosine similarity** (interpretable keyword overlap baseline)  
- **Resume category prediction** (multi-class)  
- **Shortlist / Reject** prediction (binary)  
- Optional **skill extraction + overlap**

> **Copyright © Qingyang Xiao. All rights reserved.**  
> This repository is provided for demonstration and research use. Please do not redistribute the trained artifacts or derived models without permission.

---

## Demo UI

- Left: paste **CV text**
- Right: paste **JD text**
- Click **Generate** to see results

---

## Project Structure

```
resume_app/
  app.py
  inference.py
  requirements.txt
  artifacts/
    label_classes.json
    vectorizer/         # SavedModel folder (TextVectorization)
    model_mc/           # SavedModel folder (multi-class model)
    model_bin/          # SavedModel folder (binary model)
    model_match/        # SavedModel folder (CV↔JD match model)
```

> The app expects **SavedModel folders** (not only `.keras`) for compatibility with Keras 3 / TensorFlow runtime.

---

## Quick Start

### 1) Create environment (recommended)

Using Conda (Windows/macOS/Linux):

```bash
conda create -n cvjd python=3.10 -y
conda activate cvjd
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run Streamlit

From inside `resume_app/`:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Exporting Model Artifacts from Your Notebook

This app is designed to run **inference only**. You must export these artifacts once from your training notebook (`*.ipynb`) and copy them into `resume_app/artifacts/`.

### Required artifacts

- `model_mc/` (SavedModel folder)
- `model_bin/` (SavedModel folder)
- `model_match/` (SavedModel folder)
- `vectorizer/` (SavedModel folder)
- `label_classes.json` (LabelEncoder classes)

### Notebook export snippet (recommended)

Run this in the **same notebook kernel** where models exist (do NOT run via `!python ...` because it starts a new process and won’t have your in-memory variables):

```python
import os, json, tensorflow as tf

os.makedirs("artifacts", exist_ok=True)

# Save models as TensorFlow SavedModel folders
model_mc.save("artifacts/model_mc", save_format="tf")
model_bin.save("artifacts/model_bin", save_format="tf")
model_match.save("artifacts/model_match", save_format="tf")

# Save TextVectorization as SavedModel folder
vec_model = tf.keras.Sequential([vectorizer])
vec_model.save("artifacts/vectorizer", save_format="tf")

# Save label classes
with open("artifacts/label_classes.json", "w", encoding="utf-8") as f:
    json.dump(label_enc.classes_.tolist(), f, ensure_ascii=False, indent=2)

print("✅ Export complete: ./artifacts/")
```

Then copy your exported `artifacts/` into:

```
resume_app/artifacts/
```

---

## What the Metrics Mean

### Match Probability (Model)

A 0–1 score produced by trained deep model (`model_match`) indicating how strongly the model believes the CV matches the JD.

> Note: This is typically a **model confidence score**, not necessarily a calibrated “real-world probability”.

### TF-IDF Cosine

A classic baseline similarity score (0–1) measuring keyword overlap between CV and JD using TF-IDF vectors and cosine similarity. Useful for interpretability.

---

## Notes on Keras 3 Compatibility

Keras 3 restricts `load_model()` to:

- `.keras` (Keras v3 format)
- `.h5` (legacy)

TensorFlow **SavedModel folders** can’t be loaded with `load_model()` under Keras 3.  
This project uses:

- `keras.layers.TFSMLayer(...)` for most SavedModel inference layers
- `tf.saved_model.load(...).signatures["serving_default"]` for signature-based calling (especially for multi-input models)

If atch model expects named inputs, ensure you pass inputs using the correct signature keys, e.g.:

- `jd_input`
- `resume_input`

---

## Troubleshooting

### 1) Streamlit opens but errors with missing artifacts

Make sure folder looks like:

```
resume_app/artifacts/model_mc/
resume_app/artifacts/model_bin/
resume_app/artifacts/model_match/
resume_app/artifacts/vectorizer/
resume_app/artifacts/label_classes.json
```

### 2) `IndentationError`

Python is whitespace-sensitive. Ensure function bodies are indented with 4 spaces.

### 3) `expected int32 but got int64`

If vectorizer outputs `int64` but model requires `int32`, cast before inference:

```python
tokens = tf.cast(tokens, tf.int32)
```

### 4) Streamlit caching old engine

Clear cache:

```bash
streamlit cache clear
```

Then rerun.

---

## License / Copyright

**© Qingyang Xiao. All rights reserved.**

This repository and associated artifacts may include trained models and derived assets.  
Unauthorized redistribution or commercial use is prohibited without explicit permission.

---

## Acknowledgements

Built with:

- Streamlit
- TensorFlow / Keras
- scikit-learn
