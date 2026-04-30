# Document Classification System — End-to-End Technical Report

**Project:** Document_Analyse  
**Repository:** https://github.com/Charanbtech/Document_Analyse  
**Language:** Python 3.10  
**Framework:** Flask  
**Generated:** 2026-04-29

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Dataset](#3-dataset)
4. [Text Preprocessing Pipeline](#4-text-preprocessing-pipeline)
5. [Feature Engineering — TF-IDF Vectorisation](#5-feature-engineering--tf-idf-vectorisation)
6. [Machine Learning — 6 Algorithm Comparison](#6-machine-learning--6-algorithm-comparison)
7. [Model Evaluation & Selection](#7-model-evaluation--selection)
8. [Model Persistence](#8-model-persistence)
9. [Flask Web Application & REST API](#9-flask-web-application--rest-api)
10. [Database Integration](#10-database-integration)
11. [CI/CD Pipeline](#11-cicd-pipeline)
12. [Performance Results](#12-performance-results)
13. [End-to-End Flow Diagram](#13-end-to-end-flow-diagram)

---

## 1. Project Overview

The **Document Classification System** is a production-grade machine learning application that automatically categorises text documents into one of **10 predefined topic categories**. It exposes a REST API via Flask so users can submit raw text or upload `.txt`, `.pdf`, or `.docx` files and receive instant classification results with confidence scores.

The system is fully automated with a **GitHub Actions CI/CD pipeline** that runs 6 quality checks on every push.

---

## 2. Project Structure

```
dc_ml/
├── app.py                          # Flask web application & REST API
├── requirements.txt                # Python dependencies
├── .env / .env.example             # Environment variable config
├── .gitignore
│
├── models/
│   ├── classifier.py               # Core ML pipeline (6 algorithms)
│   ├── classifier_pipeline.pkl     # Saved best model (joblib)
│   ├── class_names.pkl             # Saved class labels
│   ├── training_summary.json       # Training metrics summary
│   └── model_comparison.json       # All 6 algorithm results
│
├── utils/
│   └── preprocessor.py            # Text cleaning & file extraction
│
├── database/
│   └── db_handler.py              # MongoDB + SQLite prediction storage
│
├── data/
│   └── dataset_generator.py       # Synthetic fallback dataset
│
├── static/plots/
│   ├── algorithm_comparison.png    # Bar chart: all 6 models
│   ├── confusion_matrix.png        # Heatmap of predictions
│   ├── top_features.png            # TF-IDF feature importance
│   └── cv_scores.png               # Cross-validation fold scores
│
├── templates/
│   └── index.html                  # Web UI frontend
│
├── tests/
│   └── test_app.py                 # pytest unit tests
│
└── .github/workflows/
    └── ci.yml                      # GitHub Actions CI/CD (6 jobs)
```

---

## 3. Dataset

### 3.1 Primary Dataset — 20 Newsgroups

| Property | Detail |
|----------|--------|
| **Name** | 20 Newsgroups |
| **Source** | `sklearn.datasets.fetch_20newsgroups` |
| **Type** | Multi-class text classification |
| **Total Classes Available** | 20 newsgroup topics |
| **Classes Selected** | 10 (see below) |
| **Train Samples** | 5,765 documents |
| **Test Samples** | 3,837 documents |
| **Format** | Raw newsgroup post text (English) |

### 3.2 Selected Categories (10 of 20)

| Internal Label | Display Name | Domain |
|---------------|-------------|--------|
| `alt.atheism` | Atheism | Religion/Philosophy |
| `comp.graphics` | Computer Graphics | Technology |
| `comp.sys.ibm.pc.hardware` | PC Hardware | Technology |
| `misc.forsale` | For Sale | Commerce |
| `rec.autos` | Automobiles | Recreation |
| `rec.sport.hockey` | Hockey | Recreation/Sports |
| `sci.med` | Medicine | Science |
| `sci.space` | Space | Science |
| `soc.religion.christian` | Christianity | Religion |
| `talk.politics.guns` | Politics (Guns) | Politics |

These 10 were chosen because they span **diverse semantic domains** — technology, science, religion, politics, recreation, commerce — giving the classifier a challenging and meaningful discrimination task.

### 3.3 Data Loading Strategy

```
fetch_20newsgroups()
    ↓ success → use real dataset (5,765 train / 3,837 test)
    ↓ failure  → fallback to synthetic dataset_generator.py
                  (250 samples per class, generated text)
```

The `remove=('headers',)` flag strips **newsgroup metadata headers** so the model learns from content only — not from embedded category labels in the header.

### 3.4 Fallback — Synthetic Dataset

If internet access is unavailable, `data/dataset_generator.py` creates a synthetic dataset with domain-specific keyword-rich text paragraphs, 250 samples per class, with an 80/20 train/test split.

---

## 4. Text Preprocessing Pipeline

Every document (training, test, and live prediction) goes through the same cleaning steps in `utils/preprocessor.py`:

```
Raw Text
   │
   ▼
① Lowercase conversion          → "The GPU Card..." → "the gpu card..."
   │
   ▼
② Remove email addresses        → regex \S+@\S+ → space
   │
   ▼
③ Remove URLs                   → regex http\S+|www\.\S+ → space
   │
   ▼
④ Remove numbers                → regex \d+ → space
   │
   ▼
⑤ Remove punctuation            → str.maketrans(punctuation → spaces)
   │
   ▼
⑥ Collapse whitespace           → regex \s+ → single space
   │
   ▼
⑦ Stopword removal              → NLTK English stopwords (179 words)
   │                               Fallback: 80-word hardcoded set
   ▼
⑧ Short token filter            → drop tokens with len ≤ 2
   │
   ▼
Cleaned Text String
```

### 4.1 File Text Extraction

For uploaded documents, text is extracted before preprocessing:

| File Type | Library Used | Fallback |
|-----------|-------------|---------|
| `.txt` | Built-in `open()` | — |
| `.pdf` | PyMuPDF (`fitz`) | PyPDF2 |
| `.docx` | `python-docx` | — |

Scanned/image-only PDFs are explicitly rejected with a clear error message since OCR is not supported.

---

## 5. Feature Engineering — TF-IDF Vectorisation

Each cleaned text document is converted to a numerical vector using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

### 5.1 What is TF-IDF?

- **TF (Term Frequency):** How often a word appears in a document.
- **IDF (Inverse Document Frequency):** Penalises words common across all documents.
- **Result:** Words that are distinctive to a document score high; common words score low.

### 5.2 TF-IDF Configuration per Algorithm

| Parameter | LR / SVM / NB | RF / KNN | HistGB |
|-----------|--------------|---------|--------|
| `max_features` | 80,000 | 20,000 | 10,000 |
| `ngram_range` | (1, 2) | (1, 2) | (1, 2) |
| `sublinear_tf` | True | True | True |
| `min_df` | 2 | 2 | 2 |
| `max_df` | 0.5 | 0.5 | 0.5 |
| `strip_accents` | unicode | unicode | unicode |

- **`sublinear_tf=True`:** Applies log(1+tf) scaling, reducing the dominance of very frequent terms.
- **`ngram_range=(1,2)`:** Captures both single words and two-word phrases (bigrams) for richer context.
- **`min_df=2`:** Ignores terms appearing in fewer than 2 documents (noise reduction).
- **`max_df=0.5`:** Ignores terms appearing in more than 50% of documents (too common to be discriminative).
- Smaller `max_features` for tree-based models (RF, HistGB) because they are computationally expensive on high-dimensional sparse matrices.

---

## 6. Machine Learning — 6 Algorithm Comparison

All 6 classifiers are trained on the same preprocessed corpus under `models/classifier.py`. Each is wrapped in an `sklearn.pipeline.Pipeline` with TF-IDF.

### 6.1 Algorithm Details

#### ① Logistic Regression
- **Type:** Linear probabilistic classifier
- **Solver:** `saga` (fast, handles large sparse matrices)
- **Regularisation:** L2, `C=10.0`
- **Class Weight:** `balanced` (handles class imbalance)
- **Max Iterations:** 2,000
- **Why chosen:** Fast, interpretable, strong text baseline

#### ② Linear SVM via SGD
- **Type:** Linear SVM with Stochastic Gradient Descent
- **Loss:** `modified_huber` (enables `predict_proba`)
- **Alpha:** 1e-4 (regularisation strength)
- **Max Iterations:** 200
- **Why chosen:** State-of-the-art for text; very fast training on large corpora

#### ③ Multinomial Naive Bayes
- **Type:** Probabilistic Bayesian classifier
- **Alpha:** 0.1 (Laplace smoothing)
- **Why chosen:** Purpose-built for TF-IDF count/frequency features; extremely fast

#### ④ Random Forest
- **Type:** Ensemble of decision trees (bagging)
- **Trees:** 200 estimators
- **Max Depth:** None (full depth)
- **Class Weight:** `balanced`
- **Why chosen:** Non-linear, robust to overfitting, good reference for ensemble methods

#### ⑤ Hist Gradient Boosting
- **Type:** Histogram-based Gradient Boosted Trees (sklearn's fast XGBoost-like)
- **Iterations:** 150
- **Learning Rate:** 0.1
- **Max Depth:** 6
- **Why chosen:** Powerful boosting, handles dense features efficiently

#### ⑥ k-Nearest Neighbours
- **Type:** Instance-based lazy learner
- **k:** 7 neighbours
- **Metric:** `cosine` (ideal for TF-IDF space — angle between vectors)
- **Why chosen:** Non-parametric baseline that operates directly in the TF-IDF feature space

### 6.2 Training Flow

```
For each of the 6 algorithms:
    1. Fit Pipeline(TfidfVectorizer → Classifier) on X_train
    2. Predict on X_test
    3. Compute test accuracy
    4. Run 3-fold Stratified CV on X_train
    5. Record {algorithm, test_accuracy, cv_mean, cv_std}

Sort all results by test_accuracy descending
Winner = results[0]
Save winner to classifier_pipeline.pkl
Save full comparison to model_comparison.json
```

---

## 7. Model Evaluation & Selection

### 7.1 Metrics Used

| Metric | Description |
|--------|-------------|
| **Baseline Accuracy** | DummyClassifier (most frequent class) — lower bound |
| **Test Accuracy** | Primary selection criterion |
| **CV Mean ± Std** | 3-fold (comparison) / 5-fold (final) Stratified K-Fold |
| **Classification Report** | Per-class Precision, Recall, F1-Score |
| **Confusion Matrix** | Visual heatmap of prediction errors |

### 7.2 Stratified K-Fold Cross-Validation

```
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  ↓
Preserves class distribution in every fold
  ↓
Runs on best model for final 5-fold CV report
```

### 7.3 Visualisations Generated

| Plot | File | Description |
|------|------|-------------|
| Algorithm Comparison | `algorithm_comparison.png` | Horizontal bar chart of all 6 test accuracies |
| Confusion Matrix | `confusion_matrix.png` | 10×10 heatmap (seaborn) |
| Top Features | `top_features.png` | Top 10 TF-IDF features per class (LR/SVM only) |
| CV Scores | `cv_scores.png` | 5-fold bar chart with mean line |

### 7.4 Current Best Model Results

| Metric | Value |
|--------|-------|
| Baseline Accuracy | 10.4% |
| **Test Accuracy** | **89.55%** |
| CV Mean | 93.34% |
| CV Std | ±1.13% |
| Train Samples | 5,765 |
| Test Samples | 3,837 |
| Classes | 10 |

> The model achieved **89.55% test accuracy** vs a 10.4% baseline — an **8.6× improvement** over random guessing.

---

## 8. Model Persistence

After the best model is selected, it is serialised to disk using `joblib` for efficient reloading:

```
joblib.dump(best_pipeline, models/classifier_pipeline.pkl)   # ~19 MB
joblib.dump(class_names,   models/class_names.pkl)
json.dump(summary,         models/training_summary.json)
json.dump(comparison,      models/model_comparison.json)
```

On application startup, `app.py` calls `load_model()` which deserialises the pipeline. The full `Pipeline` object (TF-IDF + Classifier) is loaded as a single unit, so prediction requires only:

```python
pipeline.predict_proba([cleaned_text])
```

---

## 9. Flask Web Application & REST API

`app.py` serves both the web UI and a REST API.

### 9.1 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the HTML web UI |
| `GET` | `/api/status` | Model load status, classes, DB stats |
| `POST` | `/api/predict/text` | Classify raw JSON text |
| `POST` | `/api/predict/file` | Upload & classify `.txt/.pdf/.docx` |
| `GET` | `/api/history` | Recent prediction history |
| `GET` | `/api/model/comparison` | Full 6-algorithm comparison results |
| `GET` | `/api/plots/<filename>` | Serve static training plots |
| `POST` | `/api/train` | Trigger model retraining |

### 9.2 Prediction Response Format

```json
{
  "success": true,
  "predicted_class": "sci.space",
  "confidence": 0.9231,
  "top_predictions": [
    {"class": "sci.space",      "probability": 0.9231},
    {"class": "sci.med",        "probability": 0.0412},
    {"class": "comp.graphics",  "probability": 0.0187},
    ...
  ],
  "input_length": 342,
  "timestamp": "2026-04-29T10:00:00"
}
```

### 9.3 Security Hardening

| Issue | Fix |
|-------|-----|
| `debug=True` in production | Reads from `FLASK_DEBUG` env var (defaults `false`) |
| `host='0.0.0.0'` binding | Reads from `FLASK_HOST` env var (defaults `127.0.0.1`) |
| Port hardcoded | Reads from `FLASK_PORT` env var (defaults `5000`) |

---

## 10. Database Integration

`database/db_handler.py` stores every prediction for history and analytics.

### 10.1 Dual-Database Strategy

```
MongoDB URI provided?
    YES → Try MongoDB Atlas connection (timeout 3s)
             ↓ success → use MongoDB
             ↓ failure  → fall back to SQLite
    NO  → Use SQLite directly
```

### 10.2 SQLite Schema (Fallback)

```sql
CREATE TABLE predictions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    input_text        TEXT NOT NULL,        -- truncated to 500 chars
    predicted_category TEXT NOT NULL,
    confidence        REAL,
    source            TEXT DEFAULT 'text',  -- 'text' or 'file'
    timestamp         TEXT NOT NULL         -- UTC ISO format
);
```

### 10.3 Available Operations

| Method | Description |
|--------|-------------|
| `store_prediction()` | Insert a new prediction record |
| `get_recent_predictions(limit)` | Fetch N most recent records |
| `get_stats()` | Total count + breakdown by category |

---

## 11. CI/CD Pipeline

**Platform:** GitHub Actions  
**Trigger:** Every `push` or `pull_request` to `main`/`master`

### 11.1 Pipeline Architecture

```
Push to GitHub
      │
      ├─── ① Lint Code ────────────────── (flake8)
      │         Hard fail on syntax errors
      │         Soft warn on style issues
      │
      ├─── ② Security Scan ────────────── (bandit)
      │         Medium+High severity only
      │         Checks for injection, debug, binding issues
      │
      ├─── ③ Type Checking ────────────── (mypy)
      │         Informational — never blocks
      │
      └─── [① + ②  must pass]
                    │
                    ▼
             ④ Unit Tests ──────────────── (pytest)
                    │
                    ▼
             ⑤ Build Artifacts ─────────── (release.zip)
                    │
                    ▼
             ⑥ Deploy (Dry Run) ──────────  Simulates production deploy
```

### 11.2 Job Details

| Job | Tool | Blocks Next? | Purpose |
|-----|------|-------------|---------|
| ① Lint | flake8 | Yes | Catch syntax errors & style |
| ② Security | bandit `-ll -ii` | Yes | Catch medium/high vulnerabilities |
| ③ Type Check | mypy | No | Static type analysis |
| ④ Unit Tests | pytest | Yes | Verify app routes work |
| ⑤ Build | zip + upload-artifact | Yes | Package releasable artifact |
| ⑥ Deploy | download + echo | No | Simulate production deployment |

### 11.3 Tests

```python
# tests/test_app.py
def test_status_endpoint(client):
    response = client.get('/api/status')
    assert response.status_code == 200
    assert 'model_loaded' in response.get_json()
    assert 'db_stats' in response.get_json()
```

---

## 12. Performance Results

### 12.1 Training Summary (Last Run)

```json
{
  "baseline_accuracy": 0.104,
  "test_accuracy":     0.8955,
  "cv_mean":           0.9334,
  "cv_std":            0.0113,
  "num_classes":       10,
  "train_samples":     5765,
  "test_samples":      3837,
  "trained_at":        "2026-04-28T11:09:50"
}
```

### 12.2 Expected Algorithm Ranking (Typical on 20 Newsgroups)

| Rank | Algorithm | Expected Test Acc |
|------|-----------|------------------|
| 🥇 1 | Linear SVM (SGD) | ~90–92% |
| 🥈 2 | Logistic Regression | ~89–91% |
| 🥉 3 | Multinomial Naive Bayes | ~85–88% |
| 4 | Hist Gradient Boosting | ~82–86% |
| 5 | Random Forest | ~75–82% |
| 6 | k-Nearest Neighbours | ~70–78% |

> Exact rankings vary by run. The pipeline always auto-selects and saves the winner.

---

## 13. End-to-End Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING PHASE                          │
│                                                             │
│  20 Newsgroups Dataset (sklearn)                            │
│       │  5,765 train / 3,837 test / 10 classes             │
│       ▼                                                     │
│  preprocess_corpus()                                        │
│   • lowercase → strip emails/URLs/numbers/punct            │
│   • remove stopwords → filter short tokens                 │
│       ▼                                                     │
│  6× Pipeline(TfidfVectorizer → Classifier)                 │
│   • LR | SVM | NB | RF | HistGB | KNN                      │
│       ▼                                                     │
│  Evaluate each → test accuracy + 3-fold CV                 │
│       ▼                                                     │
│  Select best → save to classifier_pipeline.pkl             │
│  Save comparison → model_comparison.json                   │
│  Generate plots → static/plots/                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE PHASE                          │
│                                                             │
│  User Input (text / .txt / .pdf / .docx)                   │
│       ▼                                                     │
│  extract_text_from_file()  [if file upload]                │
│       ▼                                                     │
│  preprocess_text()                                         │
│       ▼                                                     │
│  classifier_pipeline.predict_proba()                       │
│       ▼                                                     │
│  Top-5 predictions + confidence scores                     │
│       ▼                                                     │
│  store_prediction() → MongoDB / SQLite                     │
│       ▼                                                     │
│  JSON Response → Web UI                                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     CI/CD PHASE                             │
│                                                             │
│  git push → GitHub Actions                                 │
│       ▼                                                     │
│  ① flake8 lint   ② bandit scan   ③ mypy types             │
│       ▼ (pass)        ▼ (pass)                             │
│  ④ pytest unit tests                                       │
│       ▼ (pass)                                             │
│  ⑤ Build release.zip artifact                             │
│       ▼                                                     │
│  ⑥ Deploy dry-run simulation                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Dependencies

```
scikit-learn==1.4.2     # ML algorithms, pipelines, metrics
numpy==1.26.4           # Numerical operations
pandas==2.2.2           # Data handling
nltk==3.8.1             # Stopwords
PyPDF2==3.0.1           # PDF text extraction (fallback)
python-docx==1.1.0      # DOCX text extraction
pymongo==4.7.2          # MongoDB integration
flask==3.0.3            # Web framework
flask-cors==4.0.0       # Cross-origin requests
matplotlib==3.9.0       # Plot generation
seaborn==0.13.2         # Heatmap visualisation
joblib==1.4.2           # Model serialisation
tqdm==4.66.4            # Progress bars
python-dotenv==1.0.1    # .env file loading
```

---

*Report generated for Document_Analyse — Charanbtech/Document_Analyse*
