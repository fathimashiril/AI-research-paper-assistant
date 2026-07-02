"""
==================================================
PaperLens – AI Research Paper Analysis Assistant
==================================================
Single-file Flask application that extracts,
cleans, summarises, and exports research papers.

Model  : facebook/bart-large-cnn
PDF    : PyMuPDF (fitz)
Python : 3.11+
==================================================
"""

##################################################
# IMPORTS
##################################################

import os
import re
import uuid
import json
import threading
from pathlib import Path

import fitz                                      # PyMuPDF
import torch
from flask import (
    Flask, render_template, request,
    redirect, url_for, session,
    flash, jsonify
)

from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    pipeline
)
from werkzeug.utils import secure_filename

##################################################
# CONFIGURATION
##################################################

app = Flask(__name__)
app.secret_key = "paperlens-secret-2024"

BASE_DIR            = Path(__file__).parent
UPLOAD_FOLDER       = BASE_DIR / "uploads"
ALLOWED_EXT         = {"pdf"}
MAX_CONTENT_MB      = 50

app.config["UPLOAD_FOLDER"]       = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"]  = MAX_CONTENT_MB * 1024 * 1024

# Summarisation settings
MODEL_NAME          = "facebook/bart-large-cnn"
CHUNK_TOKENS        = 800          # tokens per chunk fed to BART
CHUNK_OVERLAP       = 80           # token overlap between chunks
MAX_OUT_TOKENS      = 180          # max summary tokens per chunk
MIN_OUT_TOKENS      = 60           # min summary tokens per chunk
FINAL_MAX_TOKENS    = 400          # final merged summary max
FINAL_MIN_TOKENS    = 120          # final merged summary min

# Keyword settings removed

UPLOAD_FOLDER.mkdir(exist_ok=True)

##################################################
# JOB STORE  (in-memory, keyed by job_id)
##################################################
# Each job is a dict:
#   status  : "running" | "done" | "error"
#   result  : dict with title/meta/summaries  (when done)
#   error   : str  (when error)

JOBS: dict[str, dict] = {}

##################################################
# MODEL LOADING
##################################################

print("[PaperLens] Loading BART model …")
TOKENIZER   = BartTokenizer.from_pretrained(MODEL_NAME)
BART_MODEL  = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
BART_MODEL.eval()

print("[PaperLens] All models ready.")

##################################################
# PDF EXTRACTION
##################################################

def extract_text(pdf_path: str) -> str:
    """
    Extract plain text from every page of the PDF using PyMuPDF.
    Raises ValueError for encrypted or image-only PDFs.
    """
    doc = fitz.open(pdf_path)

    if doc.is_encrypted:
        raise ValueError("PDF is password-protected and cannot be read.")

    pages: list[str] = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    doc.close()

    if not pages:
        raise ValueError(
            "No readable text found. The PDF may be scanned. "
            "Please upload a text-based PDF."
        )
    return "\n".join(pages)


def get_page_count(pdf_path: str) -> int:
    """Return total page count of the PDF."""
    doc   = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count

##################################################
# TEXT CLEANING
##################################################

def clean_text(raw: str) -> str:
    """
    Clean raw PDF text:
    - Remove standalone page numbers
    - Repair hyphenated line-breaks
    - Collapse excess blank lines
    - Normalise whitespace
    """
    text = re.sub(r"^\s*\d+\s*$", "", raw, flags=re.MULTILINE)
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def estimate_reading_time(word_count: int, wpm: int = 200) -> str:
    """Return human-readable reading time estimate."""
    minutes = max(1, round(word_count / wpm))
    if minutes < 60:
        return f"{minutes} min"
    h, m = divmod(minutes, 60)
    return f"{h} hr {m} min" if m else f"{h} hr"


def infer_title(text: str) -> str:
    """Fallback title from first substantial line of text."""
    for line in text.splitlines():
        line = line.strip()
        if 10 < len(line) < 200:
            return line
    return "Research Paper"


def extract_title_from_pdf(pdf_path: str) -> str:
    """
    Extract paper title by finding the largest-font text span
    on the first page using PyMuPDF font-size metadata.
    Falls back to infer_title() if nothing useful is found.
    """
    try:
        doc   = fitz.open(pdf_path)
        page  = doc[0]
        blocks = page.get_text("dict")["blocks"]
        doc.close()

        spans = [
            span
            for b in blocks
            for line in b.get("lines", [])
            for span in line.get("spans", [])
            if span.get("text", "").strip()
        ]

        if not spans:
            return "Research Paper"

        max_size = max(s["size"] for s in spans)

        # Collect all spans within 2pt of the largest font
        title_parts = [
            s["text"].strip()
            for s in spans
            if s["size"] >= max_size - 2 and s["text"].strip()
        ]

        title = " ".join(title_parts).strip()

        if 5 < len(title) < 250:
            return title

    except Exception:
        pass

    return "Research Paper"

def extract_authors(text: str) -> str:
    """
    Heuristically extract author names from the first ~30 lines
    of the paper text. Looks for lines between the title and the
    Abstract heading that resemble author name patterns.
    Returns a comma-separated string, or "" if nothing found.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()][:30]
    authors = []
    for line in lines:
        if re.search(r"^(abstract|introduction|1\.?\s+intro)", line, re.I):
            break
        if re.search(r"[@{}\d\[\]©http]", line):
            continue
        if not (6 < len(line) < 80):
            continue
        if re.fullmatch(r"[A-Za-z\s,\-\.]+", line):
            authors.append(line)
        if len(authors) >= 3:
            break
    return ", ".join(authors) if authors else ""


def cleanup_old_uploads(max_age_hours: int = 2) -> None:
    """Delete uploaded files older than max_age_hours to save disk space."""
    import time
    cutoff = time.time() - max_age_hours * 3600
    for f in UPLOAD_FOLDER.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
            except Exception:
                pass


##################################################
# SUMMARISATION  (smart single-pass per section)
##################################################

def tokenize(text: str) -> list[int]:
    """Encode text to token IDs without special tokens."""
    return TOKENIZER.encode(text, add_special_tokens=False)


def decode(ids: list[int]) -> str:
    """Decode token IDs back to string."""
    return TOKENIZER.decode(ids, skip_special_tokens=True)


def chunk_tokens(token_ids: list[int],
                 size: int = CHUNK_TOKENS,
                 overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Slide a window over token_ids with overlap and return
    decoded string chunks.  Keeps chunk count low by using
    a large window, so BART runs fewer times.
    """
    chunks   = []
    start    = 0
    total    = len(token_ids)

    while start < total:
        end   = min(start + size, total)
        chunks.append(decode(token_ids[start:end]))
        if end == total:
            break
        start = end - overlap

    return chunks


def bart_summarise(text: str,
                   max_len: int = MAX_OUT_TOKENS,
                   min_len: int = MIN_OUT_TOKENS) -> str:
    """
    Run one BART summarisation pass on a single text string.
    Text is truncated to CHUNK_TOKENS if needed.
    """
    inputs = TOKENIZER(
        text,
        return_tensors = "pt",
        max_length     = CHUNK_TOKENS,
        truncation     = True,
    )
    with torch.no_grad():
        ids = BART_MODEL.generate(
            inputs["input_ids"],
            max_length           = max_len,
            min_length           = min_len,
            length_penalty       = 2.0,
            num_beams            = 4,
            early_stopping       = True,
            no_repeat_ngram_size = 3,
        )
    return TOKENIZER.decode(ids[0], skip_special_tokens=True).strip()


def multi_chunk_summarise(text: str,
                          max_len: int = FINAL_MAX_TOKENS,
                          min_len: int = FINAL_MIN_TOKENS) -> str:
    """
    Summarise arbitrarily long text:
    1. Split into overlapping chunks.
    2. Summarise each chunk individually.
    3. Merge chunk summaries.
    4. One final BART pass on the merged text.

    This keeps quality high while minimising the number of
    BART calls — each section runs BART only (n_chunks + 1) times.
    """
    token_ids = tokenize(text)

    # Short enough for a single pass?
    if len(token_ids) <= CHUNK_TOKENS:
        return bart_summarise(text, max_len, min_len)

    chunks   = chunk_tokens(token_ids)
    partials = []

    for i, chunk in enumerate(chunks):
        print(f"    chunk {i+1}/{len(chunks)} …")
        partials.append(bart_summarise(chunk))

    merged = " ".join(partials)

    # Final consolidation pass
    return bart_summarise(merged, max_len, min_len)


def split_sections(text: str) -> dict[str, str]:
    """
    Heuristically split the paper into named sections by looking
    for common academic headings.  Falls back to positional thirds
    if no headings are found.

    Returns a dict with keys:
        intro, method, results, conclusion, full
    """
    heading_map = {
        "intro"      : r"(introduction|background|overview)",
        "method"     : r"(method|approach|framework|model|architecture|proposed)",
        "results"    : r"(result|experiment|evaluation|finding|performance|analysis)",
        "conclusion" : r"(conclusion|future work|discussion|limitation)",
    }

    # Split on double-newlines (paragraph breaks)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    buckets: dict[str, list[str]] = {k: [] for k in heading_map}
    buckets["other"] = []

    current = "other"
    for para in paragraphs:
        first_line = para.splitlines()[0].lower()
        matched = False
        for key, pattern in heading_map.items():
            if re.search(pattern, first_line):
                current = key
                matched = True
                break
        buckets[current].append(para)

    def join(key: str) -> str:
        return " ".join(buckets.get(key, []))

    # If section detection worked poorly, fall back to thirds
    total_text = text
    third      = len(total_text) // 3

    return {
        "intro"      : join("intro")      or total_text[:third],
        "method"     : join("method")     or total_text[third: 2 * third],
        "results"    : join("results")    or total_text[2 * third:],
        "conclusion" : join("conclusion") or total_text[-(len(total_text) // 5):],
        "full"       : total_text,
    }


def summarise_document(text: str) -> dict[str, str]:
    """
    Generate five high-quality section summaries.

    Sections
    --------
    executive_summary  – Full paper overview
    key_contributions  – What the paper introduces
    methodology        – How the research was done
    main_findings      – Key results
    conclusion         – Closing takeaways
    """
    sections = split_sections(text)

    print("[PaperLens] → Executive summary …")
    exec_sum = multi_chunk_summarise(
        sections["full"],
        max_len = FINAL_MAX_TOKENS,
        min_len = FINAL_MIN_TOKENS,
    )

    print("[PaperLens] → Key contributions …")
    contrib = multi_chunk_summarise(
        sections["intro"],
        max_len = 200,
        min_len = 60,
    )

    print("[PaperLens] → Methodology …")
    method = multi_chunk_summarise(
        sections["method"],
        max_len = 200,
        min_len = 60,
    )

    print("[PaperLens] → Main findings …")
    findings = multi_chunk_summarise(
        sections["results"],
        max_len = 200,
        min_len = 60,
    )

    print("[PaperLens] → Conclusion …")
    conclusion = multi_chunk_summarise(
        sections["conclusion"],
        max_len = 180,
        min_len = 50,
    )

    return {
        "executive_summary" : exec_sum,
        "key_contributions" : contrib,
        "methodology"       : method,
        "main_findings"     : findings,
        "conclusion"        : conclusion,
    }

##################################################
# BACKGROUND JOB
##################################################

def run_analysis_job(job_id: str, pdf_path: str) -> None:
    """
    Full pipeline executed in a background thread so the
    HTTP response can return immediately with a loading page.
    """
    try:
        print(f"[Job {job_id}] Extracting text …")
        raw   = extract_text(pdf_path)
        clean = clean_text(raw)

        pages      = get_page_count(pdf_path)
        word_count = count_words(clean)
        read_time  = estimate_reading_time(word_count)

        # Use font-size-based title extraction, fall back to text heuristic
        title   = extract_title_from_pdf(pdf_path) or infer_title(clean)
        authors = extract_authors(clean)

        print(f"[Job {job_id}] Summarising …")
        summaries = summarise_document(clean)

        JOBS[job_id] = {
            "status"   : "done",
            "result"   : {
                "title"    : title,
                "authors"  : authors,
                "meta"     : {
                    "pages"       : pages,
                    "word_count"  : word_count,
                    "reading_time": read_time,
                },
                "summaries": summaries,
            },
        }
        print(f"[Job {job_id}] Done.")

    except Exception as exc:
        print(f"[Job {job_id}] ERROR: {exc}")
        JOBS[job_id] = {"status": "error", "error": str(exc)}



##################################################
# HELPERS
##################################################

def allowed_file(filename: str) -> bool:
    """Check file has a .pdf extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

##################################################
# FLASK ROUTES
##################################################

@app.route("/", methods=["GET"])
def index():
    """Home – upload form."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accept PDF upload, start background job, redirect to loading page.
    """
    # Clean up old uploads first
    cleanup_old_uploads()

    if "pdf_file" not in request.files:
        flash("No file part found.", "error")
        return redirect(url_for("index"))

    file = request.files["pdf_file"]

    if not file or file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Please upload a PDF file.", "error")
        return redirect(url_for("index"))

    # Save upload
    safe     = secure_filename(file.filename)
    uid      = uuid.uuid4().hex[:8]
    filename = f"{uid}_{safe}"
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(pdf_path)

    # Warn user if paper is large
    try:
        raw        = extract_text(pdf_path)
        word_count = count_words(clean_text(raw))
        if word_count > 12000:
            flash(
                f"Large paper detected (~{word_count:,} words). "
                "Analysis may take 10–15 minutes. Please keep this tab open.",
                "warning"
            )
        elif word_count > 6000:
            flash(
                f"Paper has ~{word_count:,} words. "
                "Analysis may take 5–8 minutes.",
                "info"
            )
    except Exception:
        pass

    # Create job entry and start thread
    job_id          = uuid.uuid4().hex
    JOBS[job_id]    = {"status": "running"}
    session["job_id"] = job_id

    thread = threading.Thread(
        target=run_analysis_job,
        args=(job_id, pdf_path),
        daemon=True,
    )
    thread.start()

    return redirect(url_for("loading", job_id=job_id))


@app.route("/loading/<job_id>")
def loading(job_id: str):
    """Show animated loading page while background job runs."""
    return render_template("loading.html", job_id=job_id)


@app.route("/status/<job_id>")
def status(job_id: str):
    """
    JSON endpoint polled by the loading page (meta-refresh).
    Returns {"status": "running"|"done"|"error", "error": "..."}
    """
    job = JOBS.get(job_id, {"status": "error", "error": "Job not found."})
    return jsonify({"status": job["status"],
                    "error" : job.get("error", "")})


@app.route("/result/<job_id>")
def result(job_id: str):
    """Render analysis results once job is complete."""
    job = JOBS.get(job_id)

    if not job:
        flash("Session expired. Please re-upload.", "error")
        return redirect(url_for("index"))

    if job["status"] == "running":
        return redirect(url_for("loading", job_id=job_id))

    if job["status"] == "error":
        flash(job.get("error", "Unknown error."), "error")
        return redirect(url_for("index"))

    r = job["result"]

    # Store result to a temp JSON file instead of session (avoids cookie size limit)
    result_path = UPLOAD_FOLDER / f"result_{job_id}.json"
    result_path.write_text(
        json.dumps(r, ensure_ascii=False), encoding="utf-8"
    )
    session["result_job_id"] = job_id

    return render_template(
        "result.html",
        title     = r["title"],
        authors   = r.get("authors", ""),
        meta      = r["meta"],
        summaries = r["summaries"],
    )


def load_result_from_session() -> dict | None:
    """Load result data from the temp JSON file saved at result time."""
    job_id = session.get("result_job_id")
    if not job_id:
        return None
    result_path = UPLOAD_FOLDER / f"result_{job_id}.json"
    if not result_path.exists():
        return None
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return None




##################################################
# ENTRY POINT
##################################################

@app.route("/check/<job_id>")
def check(job_id: str):
    """
    Called by the meta-refresh on the loading page every 8 s.
    Redirects to result when done, back to loading when running,
    or home with error message on failure.
    """
    job = JOBS.get(job_id, {"status": "error", "error": "Job not found."})

    if job["status"] == "done":
        return redirect(url_for("result", job_id=job_id))
    elif job["status"] == "error":
        flash(job.get("error", "An error occurred."), "error")
        return redirect(url_for("index"))
    else:
        return redirect(url_for("loading", job_id=job_id))


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
