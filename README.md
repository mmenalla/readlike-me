# 📚 Goodreads ZenML Pipeline

This project implements a data pipeline to crawl a user's "Read" bookshelf from [Goodreads](https://www.goodreads.com) and optionally enrich book data by crawling each book's page for more metadata (e.g., description, author details). The pipeline is built using **[ZenML](https://zenml.io/)** for modular, trackable execution.

---

## 🔧 Features Implemented

### ✅ Goodreads Crawler

* Crawls the "Read" shelf for a given Goodreads user.
* Extracts:

  * Book title
  * Author name
  * URL of the book
* Supports crawling additional metadata like:

  * Full author name from book page
  * Book description

### ✅ ZenML Integration

* The crawling is wrapped inside a ZenML pipeline.
* The pipeline can be executed from `run.py`.

---

## 🗂️ Project Structure

```
.
├── pipelines/
│   └── goodreads_pipeline.py     # ZenML pipeline definition
├── steps/
│   └── crawl_books_step.py       # Step to crawl Goodreads books
│   └── cenrich_book_step.py      # Step to crawl additional book details (tbd)
├── utils/
│   └── goodreads_utils.py          # Core Goodreads crawling logic
├── run.py                          # Entry point to run the pipeline
├── .env                            # Contains your Goodreads user ID
├── README.md
```

---

## 🔑 Requirements

* Python 3.12
* Install dependencies:

```bash
pip install -r requirements.txt
```

Include in `requirements.txt`:

```txt
beautifulsoup4
requests
python-dotenv
zenml
```

---

## 🛠️ Setup

1. **Clone the repo**

```bash
git clone <your-repo-url>
cd <your-repo-dir>
```

2. **Create `.env` file**

```
GOODREADS_USER_ID=...
```

3. **Initialize ZenML repo**

```bash
zenml init
```

---

## ▶️ How to Run the Pipeline

Run using `.env` default user ID:

```bash
python run.py
```

Or override via CLI:

```bash
python run.py --user_id 12345678
```
### Read from MongoDB -> embed -> store in Qdrant

```bash 
python rag-app/populate_qdrant.py
```

## RAG System

```
 curl http://localhost:6333/collections/goodreads_rag/points/scroll \
  -X POST -H 'Content-Type: application/json' \
  -d '{"limit": 1, "with_payload": true, "with_vector": true}'
```

### Fine-tuning
To use "meta-llama/Llama-3.1-8B" one must first request access from Hugging Face. 
After approval:
1. Give you token permission to use the model (from UI)
2. Set environment variable HF_TOKEN
3. Install `huggingface_hub` and login:
```bash
pip install huggingface_hub
huggingface-cli login
```