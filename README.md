# Encoder-Decoder NLP Suite
**NMIMS MPSTME · Advanced ML · Experiment 6 · Dr Ami Munshi**

| Task | Description | Model |
|------|-------------|-------|
| T1 & T2 | English → Hindi + BLEU eval | Custom LSTM Seq2Seq |
| T3 | English → Spanish | MarianMT (Helsinki-NLP) |
| T4 | Text Summarization | BART (facebook/bart-large-cnn) |

---

## Setup & Run

```bash
# 1. Install
pip install flask flask-cors torch transformers datasets sacrebleu sentencepiece

# 2. (Optional) Train En→Hi model — needs data/english_hindi.csv from Kaggle
#    https://www.kaggle.com/code/uselessnoob/english-to-hindi-machine-translation
python app.py --train-hi

# 3. Start backend
python app.py          # → http://localhost:5000

# 4. Open frontend
open index.html        # or just double-click it
```

---

## Push to GitHub (new repo)

```bash
# Requires GitHub CLI: https://cli.github.com
git init && git add -A
git commit -m "feat: Encoder-Decoder NLP Suite — Experiment 6"
gh repo create encoder-decoder-nlp --public --source=. --remote=origin --push
```

---

## API Endpoints

| Method | Endpoint | Body |
|--------|----------|------|
| GET  | `/api/health`          | — |
| POST | `/api/translate/en-hi` | `{ "text": "..." }` |
| POST | `/api/translate/en-es` | `{ "text": "..." }` |
| POST | `/api/summarize`       | `{ "text": "...", "max_length": 150 }` |
