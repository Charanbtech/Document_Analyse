# 📄 Document Classification System

End-to-end ML pipeline for automatic document categorization using NLP and Scikit-learn.

---

## 🗂️ Project Structure

```
dc_ml/
├── app.py                  # Flask web application & REST API
├── predict_cli.py          # Command-line predictor
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
│
├── models/
│   └── classifier.py       # Core ML pipeline (train, evaluate, predict)
│
├── utils/
│   └── preprocessor.py     # Text preprocessing & file extraction
│
├── database/
│   └── db_handler.py       # MongoDB + SQLite storage
│
├── templates/
│   └── index.html          # Web UI
│
├── static/
│   └── plots/              # Generated visualisation PNGs
│
└── logs/
    └── training.log        # Training logs
```

---

## ⚙️ Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Configure MongoDB
cp .env.example .env
# Edit .env and add your MONGO_URI

# 3. Train the model
python models/classifier.py

# 4. Run the web app
python app.py
```

Open http://localhost:5000 in your browser.

---

## 🚀 Usage

### Web Interface
- Navigate to `http://localhost:5000`
- Paste text or upload a `.txt`, `.pdf`, or `.docx` file
- Click **Classify** to get instant predictions with confidence scores
- View prediction history and analytics plots

### CLI
```bash
# Classify text
python predict_cli.py --text "NASA discovered new exoplanets using the James Webb telescope"

# Classify a file
python predict_cli.py --file report.pdf
```

### REST API
```bash
# Text prediction
curl -X POST http://localhost:5000/api/predict/text \
  -H "Content-Type: application/json" \
  -d '{"text": "The hockey playoffs overtime game was incredible"}'

# File prediction
curl -X POST http://localhost:5000/api/predict/file \
  -F "file=@document.pdf"

# Check status
curl http://localhost:5000/api/status

# Prediction history
curl http://localhost:5000/api/history
```

---

## 📊 Model Details

| Component       | Detail                              |
|----------------|--------------------------------------|
| Dataset        | 20 Newsgroups (10 categories, subset)|
| Features       | TF-IDF (uni+bigrams, 10k–30k vocab)  |
| Classifier     | Logistic Regression (L2, balanced)   |
| Tuning         | GridSearchCV (3-fold)                |
| Validation     | StratifiedKFold (5 splits)           |
| Baseline       | DummyClassifier (most_frequent)      |
| Storage        | SQLite (default) / MongoDB Atlas     |

### Expected Results
- Baseline Accuracy: ~10%
- Test Accuracy: ~85–90%
- CV Accuracy: ~85–90% ± ~2%

---

## 🔮 Future Enhancements
- BERT / DistilBERT transformer fine-tuning
- Streamlit dashboard alternative
- Docker containerization
- CI/CD with GitHub Actions
- Domain-specific dataset support
