"""
Flask Web Application for Document Classification System
Provides REST API endpoints and serves the web UI.
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# ── Setup paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.classifier import predict, load_model, MODEL_PATH, MODELS_DIR
from utils.preprocessor import extract_text_from_file, preprocess_text
from database.db_handler import DatabaseHandler

# ── App init ───────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Load model at startup ──────────────────────────────────────────────────────
model = None
class_names = []

def get_model():
    global model, class_names
    if model is None:
        try:
            model, class_names = load_model()
            logger.info("Model loaded successfully")
        except FileNotFoundError:
            logger.warning("Model not found. Please run training first.")
    return model, class_names

# ── Database ───────────────────────────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", None)
db = DatabaseHandler(mongo_uri=MONGO_URI)

# ── Load training summary ──────────────────────────────────────────────────────
def load_summary():
    summary_path = MODELS_DIR / "training_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {}

# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def status():
    """Model and system status."""
    m, cn = get_model()
    summary = load_summary()
    return jsonify({
        "model_loaded": m is not None,
        "num_classes": len(cn),
        "classes": cn,
        "summary": summary,
        "db_stats": db.get_stats()
    })


@app.route('/api/predict/text', methods=['POST'])
def predict_text():
    """Classify raw text input."""
    data = request.get_json(force=True)
    text = data.get('text', '').strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    m, cn = get_model()
    if m is None:
        return jsonify({"error": "Model not loaded. Train the model first."}), 503

    try:
        result = predict(text, model=m, class_names=cn)
        db.store_prediction(text, result['predicted_class'], result['confidence'], source='text')
        return jsonify({
            "success": True,
            "predicted_class": result['predicted_class'],
            "confidence": result['confidence'],
            "top_predictions": result['top_predictions'],
            "input_length": len(text),
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict/file', methods=['POST'])
def predict_file():
    """Classify an uploaded document file."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    allowed = {'.txt', '.pdf', '.docx'}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type: {ext}. Use .txt, .pdf, or .docx"}), 400

    m, cn = get_model()
    if m is None:
        return jsonify({"error": "Model not loaded. Train the model first."}), 503

    try:
        fd, tmp_name = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        try:
            file.save(tmp_name)
            text = extract_text_from_file(tmp_name)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

        if not text.strip():
            return jsonify({"error": "Could not extract text. If this is a scanned PDF (image-only), please note that OCR is not supported."}), 422

        result = predict(text, model=m, class_names=cn)
        db.store_prediction(text, result['predicted_class'], result['confidence'], source='file')

        return jsonify({
            "success": True,
            "filename": file.filename,
            "predicted_class": result['predicted_class'],
            "confidence": result['confidence'],
            "top_predictions": result['top_predictions'],
            "extracted_length": len(text),
            "preview": text[:300] + "..." if len(text) > 300 else text,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"File prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/history')
def history():
    """Recent prediction history."""
    limit = int(request.args.get('limit', 20))
    records = db.get_recent_predictions(limit=limit)
    return jsonify({"predictions": records, "count": len(records)})


@app.route('/api/model/comparison')
def model_comparison():
    """Return the multi-algorithm comparison results."""
    comparison_path = MODELS_DIR / "model_comparison.json"
    if comparison_path.exists():
        with open(comparison_path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No comparison data found. Train the model first."}), 404


@app.route('/api/plots/<filename>')
def serve_plot(filename):
    plots_dir = PROJECT_ROOT / "static" / "plots"
    return send_from_directory(str(plots_dir), filename)


@app.route('/api/train', methods=['POST'])
def train():
    """Trigger model training (runs synchronously for simplicity)."""
    try:
        from models.classifier import train_pipeline
        global model, class_names
        model, class_names, summary = train_pipeline()
        return jsonify({"success": True, "summary": summary})
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    get_model()  # Pre-load
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '127.0.0.1')  # nosec B104
    port = int(os.environ.get('FLASK_PORT', '5000'))
    app.run(debug=debug_mode, host=host, port=port)
