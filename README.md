# AI-research-paper-assistant
# PaperLens – AI Research Paper Analyzer

## Overview

PaperLens is an AI-powered web application that analyzes research papers in PDF format. It extracts text from uploaded PDFs, identifies key metadata, and generates structured summaries using the BART (facebook/bart-large-cnn) model.

## Features

* Upload research papers in PDF format
* Automatic text extraction
* Title and author extraction
* Page count, word count, and estimated reading time
* Section-wise summarization
* Executive Summary generation
* Background processing using Flask
* Simple and responsive web interface

## Technologies Used

* Python
* Flask
* Hugging Face Transformers
* BART (facebook/bart-large-cnn)
* PyTorch
* PyMuPDF (fitz)
* HTML
* CSS

## Project Workflow

```text
Upload PDF
      │
      ▼
Extract Text
      │
      ▼
Clean Text
      │
      ▼
Extract Metadata
      │
      ▼
Split into Sections
      │
      ▼
Generate Summaries
      │
      ▼
Display Results
```

## Project Structure

```text
PaperLens/
│
├── app.py
├── uploads/
├── templates/
├── static/
├── requirements.txt
└── README.md
```

## Future Improvements

* RAG-based question answering
* Keyword extraction
* Citation analysis
* Multi-language support

# Home Page :
<img width="940" height="464" alt="image" src="https://github.com/user-attachments/assets/eaabece3-1601-4095-a2ce-87636188dbcc" />

# Loading Page:
<img width="940" height="442" alt="image" src="https://github.com/user-attachments/assets/7f879cb1-e7d3-40fb-b2b9-d87bc940fd59" />
# Result :
<img width="940" height="442" alt="image" src="https://github.com/user-attachments/assets/7231a598-714d-4cba-a5b7-f7fa02018b6e" />
<img width="940" height="444" alt="image" src="https://github.com/user-attachments/assets/7c7dcaf4-fe77-4b08-a6ed-79323993dcaa" />












