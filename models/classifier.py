"""
Document Classification ML Pipeline
Trains 6 algorithms, picks the best by test accuracy, and saves it.

Algorithms compared:
  1. Logistic Regression
  2. Linear SVM (SGD)
  3. Multinomial Naive Bayes
  4. Random Forest
  5. Gradient Boosting (HistGradientBoosting)
  6. k-Nearest Neighbours
"""

import os
import sys
import json
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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix
)

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_DIR     = PROJECT_ROOT / "data"
STATIC_DIR   = PROJECT_ROOT / "static"
LOGS_DIR     = PROJECT_ROOT / "logs"

for d in (MODELS_DIR, DATA_DIR, STATIC_DIR / "plots", LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATH      = MODELS_DIR / "classifier_pipeline.pkl"
CLASSES_PATH    = MODELS_DIR / "class_names.pkl"
COMPARISON_PATH = MODELS_DIR / "model_comparison.json"

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

# ── 20 Newsgroups categories ───────────────────────────────────────────────────
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

# ── Class-name display mapping ─────────────────────────────────────────────────
CLASS_MAP = {
    'alt.atheism':             'Atheism',
    'comp.graphics':           'Computer Graphics',
    'comp.sys.ibm.pc.hardware':'PC Hardware',
    'misc.forsale':            'For Sale',
    'rec.autos':               'Automobiles',
    'rec.sport.hockey':        'Hockey',
    'sci.med':                 'Medicine',
    'sci.space':               'Space',
    'soc.religion.christian':  'Christianity',
    'talk.politics.guns':      'Politics (Guns)',
}

# ── Pre-processing helper ──────────────────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT))
from utils.preprocessor import preprocess_text


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

class DataBunch:
    """Simple container mimicking sklearn Bunch."""
    def __init__(self, data, target, target_names):
        self.data         = data
        self.target       = target
        self.target_names = target_names


def load_data():
    """Download / load 20 Newsgroups, falling back to synthetic dataset."""
    logger.info("Loading dataset …")
    try:
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

    sys.path.insert(0, str(DATA_DIR))
    from dataset_generator import generate_dataset
    X_tr, X_te, y_tr, y_te, cats = generate_dataset(samples_per_class=250)
    train = DataBunch(data=X_tr, target=np.array(y_tr), target_names=cats)
    test  = DataBunch(data=X_te, target=np.array(y_te), target_names=cats)
    logger.info(f"Train samples : {len(train.data)}  (synthetic)")
    logger.info(f"Test  samples : {len(test.data)}   (synthetic)")
    return train, test


def preprocess_corpus(corpus):
    """Apply text preprocessing to a list of documents."""
    logger.info("Preprocessing corpus …")
    return [preprocess_text(doc) for doc in corpus]


# ══════════════════════════════════════════════════════════════════════════════
# SIX ALGORITHM DEFINITIONS
# Each entry: (display_name, classifier_object, needs_dense_matrix)
# ══════════════════════════════════════════════════════════════════════════════

def get_candidate_pipelines():
    """
    Returns a list of (name, Pipeline) tuples — one per algorithm.
    All share the same TF-IDF vectoriser settings for a fair comparison.
    """
    # Shared TF-IDF base settings
    tfidf_base = dict(
        sublinear_tf=True,
        min_df=2,
        max_df=0.5,
        strip_accents='unicode',
        ngram_range=(1, 2),
    )
    # Separate configs with different max_features budgets
    tfidf_large  = dict(**tfidf_base, max_features=80_000)   # LR, SGD, MNB
    tfidf_medium = dict(**tfidf_base, max_features=20_000)   # RF, KNN
    tfidf_small  = dict(**tfidf_base, max_features=10_000)   # HistGB

    candidates = [
        (
            "Logistic Regression",
            Pipeline([
                ('tfidf', TfidfVectorizer(**tfidf_large)),
                ('clf',   LogisticRegression(
                    solver='saga', C=10.0, max_iter=2000,
                    class_weight='balanced', random_state=42
                ))
            ])
        ),
        (
            "Linear SVM (SGD)",
            Pipeline([
                ('tfidf', TfidfVectorizer(**tfidf_large)),
                ('clf',   SGDClassifier(
                    loss='modified_huber',        # supports predict_proba
                    alpha=1e-4, max_iter=200,
                    class_weight='balanced',
                    random_state=42, n_jobs=-1
                ))
            ])
        ),
        (
            "Multinomial Naive Bayes",
            Pipeline([
                ('tfidf', TfidfVectorizer(**tfidf_large)),
                ('clf',   MultinomialNB(alpha=0.1))
            ])
        ),
        (
            "Random Forest",
            Pipeline([
                ('tfidf', TfidfVectorizer(**tfidf_medium)),
                ('clf',   RandomForestClassifier(
                    n_estimators=200, max_depth=None,
                    class_weight='balanced', random_state=42, n_jobs=-1
                ))
            ])
        ),
        (
            "Hist Gradient Boosting",
            Pipeline([
                ('tfidf', TfidfVectorizer(**tfidf_small)),
                ('clf',   HistGradientBoostingClassifier(
                    max_iter=150, learning_rate=0.1,
                    max_depth=6, random_state=42
                ))
            ])
        ),
        (
            "k-Nearest Neighbours",
            Pipeline([
                ('tfidf', TfidfVectorizer(**tfidf_medium)),
                ('clf',   KNeighborsClassifier(n_neighbors=7, metric='cosine', n_jobs=-1))
            ])
        ),
    ]
    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN & COMPARE
# ══════════════════════════════════════════════════════════════════════════════

def train_and_compare(X_train, y_train, X_test, y_test):
    """
    Train every candidate pipeline on (X_train, y_train),
    evaluate on (X_test, y_test), and return:
      - results: list[dict] sorted best → worst
      - best_name: str
      - best_pipeline: fitted Pipeline
    """
    candidates  = get_candidate_pipelines()
    results     = []
    best_acc    = -1.0
    best_name   = None
    best_model  = None

    logger.info("=" * 60)
    logger.info("  Comparing 6 Algorithms")
    logger.info("=" * 60)

    for name, pipeline in candidates:
        logger.info(f"\n▶  Training: {name}")
        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)

            # 3-fold CV on training set (fast)
            cv     = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_sc  = cross_val_score(pipeline, X_train, y_train,
                                     cv=cv, scoring='accuracy', n_jobs=-1)

            logger.info(f"   Test Accuracy : {acc:.4f}")
            logger.info(f"   CV  Accuracy  : {cv_sc.mean():.4f} ± {cv_sc.std():.4f}")

            results.append({
                "algorithm":    name,
                "test_accuracy":round(acc, 4),
                "cv_mean":      round(float(cv_sc.mean()), 4),
                "cv_std":       round(float(cv_sc.std()),  4),
            })

            if acc > best_acc:
                best_acc   = acc
                best_name  = name
                best_model = pipeline

        except Exception as e:
            logger.error(f"   {name} failed: {e}")
            results.append({
                "algorithm":    name,
                "test_accuracy": None,
                "cv_mean":       None,
                "cv_std":        None,
                "error":         str(e),
            })

    results.sort(key=lambda r: r["test_accuracy"] or -1, reverse=True)

    logger.info("\n" + "=" * 60)
    logger.info("  Algorithm Comparison Results")
    logger.info("=" * 60)
    for rank, r in enumerate(results, 1):
        logger.info(
            f"  #{rank}  {r['algorithm']:<30} "
            f"Test={r['test_accuracy']:.4f}  "
            f"CV={r['cv_mean']:.4f}±{r['cv_std']:.4f}"
        )
    logger.info(f"\n  🏆  Best Model: {best_name}  (acc={best_acc:.4f})")
    logger.info("=" * 60)

    return results, best_name, best_model


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_test, class_names):
    """Full classification report for the best model."""
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    logger.info(f"\nTest Accuracy : {acc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=class_names))
    return acc, y_pred


def baseline_accuracy(X_train, y_train, X_test, y_test):
    """DummyClassifier baseline."""
    dummy = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy.fit(X_train, y_train)
    acc = accuracy_score(y_test, dummy.predict(X_test))
    logger.info(f"Baseline Acc  : {acc:.4f}")
    return acc


def get_readable_names(class_names):
    return [CLASS_MAP.get(c, c) for c in class_names]


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_algorithm_comparison(results):
    """Horizontal bar chart comparing all 6 algorithms."""
    names = [r["algorithm"] for r in results]
    accs  = [r["test_accuracy"] or 0 for r in results]
    errs  = [r["cv_std"] or 0 for r in results]

    colors = ["#2ECC71" if i == 0 else "#3498DB" for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names[::-1], accs[::-1], xerr=errs[::-1],
                   color=colors[::-1], edgecolor='white', linewidth=0.8,
                   error_kw=dict(ecolor='gray', capsize=4))
    ax.set_xlabel("Test Accuracy", fontsize=12)
    ax.set_title("Algorithm Comparison — Document Classification", fontsize=14, pad=12)
    ax.set_xlim(0, 1.05)

    for bar, acc in zip(bars, accs[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{acc:.4f}", va='center', fontsize=9)

    plt.tight_layout()
    path = STATIC_DIR / "plots" / "algorithm_comparison.png"
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info(f"Saved algorithm comparison → {path}")


def plot_confusion_matrix(y_test, y_pred, class_names):
    readable = get_readable_names(class_names)
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=readable, yticklabels=readable, ax=ax)
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
    """Bar charts of top TF-IDF features per class (LR / SGD only)."""
    clf = model.named_steps['clf']
    if not hasattr(clf, 'coef_'):
        logger.info("Skipping top-features plot (model has no coef_).")
        return

    readable = get_readable_names(class_names)
    tfidf    = model.named_steps['tfidf']
    vocab    = np.array(tfidf.get_feature_names_out())

    n_classes = len(class_names)
    cols, rows = 2, (n_classes + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = axes.flatten()

    for i, (name, coef) in enumerate(zip(readable, clf.coef_)):
        top_idx   = np.argsort(coef)[-top_n:]
        axes[i].barh(vocab[top_idx], coef[top_idx], color='steelblue')
        axes[i].set_title(name, fontsize=12)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Top Discriminative Features per Class', fontsize=16, y=1.01)
    plt.tight_layout()
    path = STATIC_DIR / "plots" / "top_features.png"
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved top features plot → {path}")


def plot_cv_scores(cv_scores, best_name):
    fig, ax = plt.subplots(figsize=(7, 4))
    folds  = [f"Fold {i+1}" for i in range(len(cv_scores))]
    colors = ['#4C9ED9' if s >= cv_scores.mean() else '#E07B54' for s in cv_scores]
    bars   = ax.bar(folds, cv_scores, color=colors, edgecolor='white', linewidth=0.8)
    ax.axhline(cv_scores.mean(), color='gray', linestyle='--', linewidth=1.2,
               label=f'Mean = {cv_scores.mean():.3f}')
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'5-Fold CV Scores — {best_name}')
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
    logger.info(f"Best model saved → {MODEL_PATH}")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run train_pipeline() first.")
    model       = joblib.load(MODEL_PATH)
    class_names = joblib.load(CLASSES_PATH)
    return model, class_names


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict(text: str, model=None, class_names=None):
    """
    Predict the category of a text document.
    Returns dict with: predicted_class, confidence, top_predictions
    """
    if model is None:
        model, class_names = load_model()

    cleaned = preprocess_text(text)

    # Use predict_proba if available, else decision_function → softmax
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba([cleaned])[0]
    else:
        scores = model.decision_function([cleaned])[0]
        proba  = np.exp(scores) / np.exp(scores).sum()

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
    logger.info("  Document Classification System — Multi-Algorithm Training")
    logger.info("=" * 60)

    # ── 1. Data ────────────────────────────────────────────────────────────────
    train_data, test_data = load_data()
    X_train = preprocess_corpus(train_data.data)
    X_test  = preprocess_corpus(test_data.data)
    y_train, y_test = train_data.target, test_data.target
    class_names = list(train_data.target_names)

    # ── 2. Baseline ────────────────────────────────────────────────────────────
    base_acc = baseline_accuracy(X_train, y_train, X_test, y_test)

    # ── 3. Train & compare 6 algorithms ───────────────────────────────────────
    results, best_name, best_model = train_and_compare(
        X_train, y_train, X_test, y_test
    )

    # ── 4. Save comparison report ──────────────────────────────────────────────
    with open(COMPARISON_PATH, 'w') as f:
        json.dump({"algorithms": results, "winner": best_name}, f, indent=2)
    logger.info(f"Comparison report saved → {COMPARISON_PATH}")

    # ── 5. Full evaluation of best model ──────────────────────────────────────
    test_acc, y_pred = evaluate(best_model, X_test, y_test, class_names)

    # ── 6. 5-fold CV on best model ────────────────────────────────────────────
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_sc  = cross_val_score(best_model, X_train, y_train,
                             cv=cv, scoring='accuracy', n_jobs=-1)
    logger.info(f"CV Accuracy   : {cv_sc.mean():.4f} ± {cv_sc.std():.4f}")

    # ── 7. Plots ───────────────────────────────────────────────────────────────
    plot_algorithm_comparison(results)
    plot_confusion_matrix(y_test, y_pred, class_names)
    plot_top_features(best_model, class_names)
    plot_cv_scores(cv_sc, best_name)

    # ── 8. Save best model ─────────────────────────────────────────────────────
    save_model(best_model, class_names)

    # ── 9. Summary ─────────────────────────────────────────────────────────────
    summary = {
        "best_algorithm":   best_name,
        "baseline_accuracy":round(base_acc, 4),
        "test_accuracy":    round(test_acc, 4),
        "cv_mean":          round(float(cv_sc.mean()), 4),
        "cv_std":           round(float(cv_sc.std()),  4),
        "num_classes":      len(class_names),
        "train_samples":    len(X_train),
        "test_samples":     len(X_test),
        "algorithms_tried": [r["algorithm"] for r in results],
        "trained_at":       datetime.utcnow().isoformat(),
    }

    with open(MODELS_DIR / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info(f"  Winner       : {best_name}")
    logger.info(f"  Baseline     : {base_acc:.2%}")
    logger.info(f"  Test Acc     : {test_acc:.2%}")
    logger.info(f"  CV Acc       : {cv_sc.mean():.2%} ± {cv_sc.std():.2%}")
    logger.info("=" * 60)

    return best_model, class_names, summary


if __name__ == "__main__":
    train_pipeline()
