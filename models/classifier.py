"""
Document Classification ML Pipeline
Trains, evaluates, and serves a TF-IDF + Logistic Regression classifier
on the 20 Newsgroups dataset with GridSearchCV optimization.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import datetime

from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_DIR     = PROJECT_ROOT / "data"
STATIC_DIR   = PROJECT_ROOT / "static"
LOGS_DIR     = PROJECT_ROOT / "logs"

for d in (MODELS_DIR, DATA_DIR, STATIC_DIR / "plots", LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATH   = MODELS_DIR / "classifier_pipeline.pkl"
CLASSES_PATH = MODELS_DIR / "class_names.pkl"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "training.log")
    ]
)
logger = logging.getLogger(__name__)

# ── 20 Newsgroups categories (subset for meaningful demo) ──────────────────────
CATEGORIES = [
    'alt.atheism',
    'comp.graphics',
    'comp.sys.ibm.pc.hardware',
    'misc.forsale',
    'rec.autos',
    'rec.sport.hockey',
    'sci.med',
    'sci.space',
    'soc.religion.christian',
    'talk.politics.guns',
]

# ── Preprocessing helper ───────────────────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT))
from utils.preprocessor import preprocess_text


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

class DataBunch:
    """Simple container mimicking sklearn Bunch."""
    def __init__(self, data, target, target_names):
        self.data = data
        self.target = target
        self.target_names = target_names


def load_data():
    """Download / load 20 Newsgroups, falling back to synthetic dataset."""
    logger.info("Loading dataset …")

    # Try official 20 Newsgroups first
    try:
        # Removing only 'headers' prevents the model from explicitly reading the category label from metadata,
        # but retaining 'footers' and 'quotes' provides massive contextual hints, dramatically increasing accuracy.
        remove_fields = ('headers',)
        train = fetch_20newsgroups(
            subset='train', categories=CATEGORIES,
            remove=remove_fields, random_state=42
        )
        test = fetch_20newsgroups(
            subset='test', categories=CATEGORIES,
            remove=remove_fields, random_state=42
        )
        logger.info(f"Train samples : {len(train.data)}")
        logger.info(f"Test  samples : {len(test.data)}")
        logger.info(f"Classes       : {train.target_names}")
        return train, test
    except Exception as e:
        logger.warning(f"20 Newsgroups download failed ({e}). Using synthetic dataset.")

    # Fallback: synthetic dataset
    sys.path.insert(0, str(DATA_DIR))
    from dataset_generator import generate_dataset

    X_tr, X_te, y_tr, y_te, cats = generate_dataset(samples_per_class=250)
    train = DataBunch(data=X_tr, target=np.array(y_tr), target_names=cats)
    test  = DataBunch(data=X_te, target=np.array(y_te), target_names=cats)

    logger.info(f"Train samples : {len(train.data)}  (synthetic)")
    logger.info(f"Test  samples : {len(test.data)}   (synthetic)")
    logger.info(f"Classes       : {train.target_names}")
    return train, test


def preprocess_corpus(corpus):
    """Apply text preprocessing to a list of documents."""
    logger.info("Preprocessing corpus …")
    return [preprocess_text(doc) for doc in corpus]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline() -> Pipeline:
    """Construct the TF-IDF → LogisticRegression pipeline."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            sublinear_tf=True,
            min_df=2,
            max_df=0.5,
            strip_accents='unicode'
        )),
        ('clf', LogisticRegression(
            solver='saga',
            max_iter=2000,
            class_weight='balanced',
            random_state=42
        ))
    ])


def tune_pipeline(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """
    Run GridSearchCV to find the best TF-IDF + LR hyperparameters.
    Returns the best estimator fitted on the full training set.
    """
    param_grid = {
        'tfidf__max_features': [50_000, 100_000, None],
        'tfidf__ngram_range':  [(1, 2)],
        'clf__C':              [5.0, 10.0, 25.0],
        'clf__penalty':        ['l2']
    }

    logger.info("Starting GridSearchCV (this may take a few minutes) …")
    gs = GridSearchCV(
        pipeline, param_grid,
        cv=3, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    gs.fit(X_train, y_train)

    logger.info(f"Best params   : {gs.best_params_}")
    logger.info(f"Best CV score : {gs.best_score_:.4f}")
    return gs.best_estimator_


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_test, class_names):
    """Print metrics and return accuracy."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"\nTest Accuracy : {acc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=class_names))
    return acc, y_pred


def cross_validate(model, X, y):
    """5-fold stratified cross-validation."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    logger.info(f"CV Accuracy   : {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


def baseline_accuracy(X_train, y_train, X_test, y_test):
    """DummyClassifier baseline."""
    dummy = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy.fit(X_train, y_train)
    acc = accuracy_score(y_test, dummy.predict(X_test))
    logger.info(f"Baseline Acc  : {acc:.4f}")
    return acc


# ── Class Name Mapping ──
CLASS_MAP = {
    'alt.atheism': 'Atheism',
    'comp.graphics': 'Computer Graphics',
    'comp.sys.ibm.pc.hardware': 'PC Hardware',
    'misc.forsale': 'For Sale',
    'rec.autos': 'Automobiles',
    'rec.sport.hockey': 'Hockey',
    'sci.med': 'Medicine',
    'sci.space': 'Space',
    'soc.religion.christian': 'Christianity',
    'talk.politics.guns': 'Politics (Guns)'
}

def get_readable_names(class_names):
    return [CLASS_MAP.get(c, c) for c in class_names]


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_test, y_pred, class_names):
    readable_names = get_readable_names(class_names)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=readable_names, yticklabels=readable_names, ax=ax)
    ax.set_title('Confusion Matrix', fontsize=14, pad=12)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    path = STATIC_DIR / "plots" / "confusion_matrix.png"
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info(f"Saved confusion matrix → {path}")


def plot_top_features(model, class_names, top_n=10):
    """Bar charts of top TF-IDF features per class."""
    readable_names = get_readable_names(class_names)
    tfidf = model.named_steps['tfidf']
    clf   = model.named_steps['clf']
    vocab = np.array(tfidf.get_feature_names_out())

    n_classes = len(class_names)
    cols = 2
    rows = (n_classes + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = axes.flatten()

    for i, (name, coef) in enumerate(zip(readable_names, clf.coef_)):
        top_idx  = np.argsort(coef)[-top_n:]
        top_words = vocab[top_idx]
        top_coefs = coef[top_idx]
        axes[i].barh(top_words, top_coefs, color='steelblue')
        axes[i].set_title(name, fontsize=12)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Top Discriminative Features per Class', fontsize=16, y=1.01)
    plt.tight_layout()
    path = STATIC_DIR / "plots" / "top_features.png"
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved top features plot → {path}")


def plot_cv_scores(cv_scores):
    fig, ax = plt.subplots(figsize=(7, 4))
    folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
    colors = ['#4C9ED9' if s >= cv_scores.mean() else '#E07B54' for s in cv_scores]
    bars = ax.bar(folds, cv_scores, color=colors, edgecolor='white', linewidth=0.8)
    ax.axhline(cv_scores.mean(), color='gray', linestyle='--', linewidth=1.2,
               label=f'Mean = {cv_scores.mean():.3f}')
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('Accuracy')
    ax.set_title('5-Fold Cross-Validation Scores')
    ax.legend()
    for bar, score in zip(bars, cv_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    path = STATIC_DIR / "plots" / "cv_scores.png"
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info(f"Saved CV scores plot → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model, class_names):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(class_names, CLASSES_PATH)
    logger.info(f"Model saved → {MODEL_PATH}")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run train_pipeline() first.")
    model      = joblib.load(MODEL_PATH)
    class_names = joblib.load(CLASSES_PATH)
    return model, class_names


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict(text: str, model=None, class_names=None):
    """
    Predict the category of a text document.

    Returns:
        dict with keys: predicted_class, confidence, top_predictions
    """
    if model is None:
        model, class_names = load_model()

    cleaned = preprocess_text(text)
    proba   = model.predict_proba([cleaned])[0]
    top_idx = np.argsort(proba)[::-1]

    return {
        "predicted_class": class_names[top_idx[0]],
        "confidence":      float(proba[top_idx[0]]),
        "top_predictions": [
            {"class": class_names[i], "probability": float(proba[i])}
            for i in top_idx[:5]
        ]
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FLOW
# ══════════════════════════════════════════════════════════════════════════════

def train_pipeline():
    logger.info("=" * 60)
    logger.info("  Document Classification System — Training")
    logger.info("=" * 60)

    # 1. Data
    train_data, test_data = load_data()
    X_train_raw = preprocess_corpus(train_data.data)
    X_test_raw  = preprocess_corpus(test_data.data)
    y_train, y_test = train_data.target, test_data.target
    class_names = list(train_data.target_names)

    # 2. Baseline
    base_acc = baseline_accuracy(X_train_raw, y_train, X_test_raw, y_test)

    # 3. Build + tune pipeline
    pipeline = build_pipeline()
    best_model = tune_pipeline(pipeline, X_train_raw, y_train)

    # 4. Evaluate
    test_acc, y_pred = evaluate(best_model, X_test_raw, y_test, class_names)

    # 5. Cross-validation
    cv_scores = cross_validate(best_model, X_train_raw, y_train)

    # 6. Plots
    plot_confusion_matrix(y_test, y_pred, class_names)
    plot_top_features(best_model, class_names)
    plot_cv_scores(cv_scores)

    # 7. Save
    save_model(best_model, class_names)

    summary = {
        "baseline_accuracy": round(base_acc, 4),
        "test_accuracy":     round(test_acc, 4),
        "cv_mean":           round(float(cv_scores.mean()), 4),
        "cv_std":            round(float(cv_scores.std()), 4),
        "num_classes":       len(class_names),
        "train_samples":     len(X_train_raw),
        "test_samples":      len(X_test_raw),
        "trained_at":        datetime.utcnow().isoformat()
    }

    # Persist summary
    import json
    with open(MODELS_DIR / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info(f"  Baseline : {base_acc:.2%}")
    logger.info(f"  Test Acc : {test_acc:.2%}")
    logger.info(f"  CV Acc   : {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
    logger.info("=" * 60)

    return best_model, class_names, summary


if __name__ == "__main__":
    train_pipeline()
