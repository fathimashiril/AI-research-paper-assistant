# 🔬 PaperLens – AI Research Paper Analysis Assistant

Final Year B.Tech Artificial Intelligence & Data Science Project

PaperLens is a locally-running Flask web app that analyses research papers.
It uses **facebook/bart-large-cnn** for summarisation — fully offline, no external APIs.

---

## Features
- Upload any text-based PDF research paper
- Extracts 5 structured sections: Executive Summary, Key Contributions,
  Methodology, Main Findings, Conclusion
- Author name extraction from first page
- Smart title extraction using PDF font-size metadata
- Word count warning for large papers before analysis starts
- Auto-cleanup of uploaded files older than 2 hours
- Copy-to-clipboard button on every summary section
- Live loading page — browser never times out
- Download report as TXT or PDF
- Professional academic UI (pure HTML/CSS, no JS frameworks)

---

## Setup (Python 3.11)

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate (Windows)
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python app.py
```

Open browser at: **http://127.0.0.1:5000**

> First run downloads the BART model (~1.6 GB). Subsequent runs are instant.

---

## Project Structure

```
PaperLens/
├── app.py                  # Complete Flask application
├── requirements.txt
├── uploads/                # Auto-created at runtime
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── loading.html
│   └── result.html
└── static/
    └── style.css
```

---

## Tech Stack

| Component     | Technology                        |
|---------------|-----------------------------------|
| Backend       | Python 3.11, Flask 3.x            |
| PDF Parsing   | PyMuPDF                           |
| Summarisation | facebook/bart-large-cnn (HF)      |
| PDF Export    | FPDF2                             |
| Frontend      | HTML5 + CSS3 (Jinja2)             |

---

## Changelog (v3 – No Keywords)

- Removed KeyBERT keyword extraction entirely
- Removed sentence-transformers dependency
- Cleaner, faster startup (one model instead of two)
- All keyword UI, CSS, and export references removed
