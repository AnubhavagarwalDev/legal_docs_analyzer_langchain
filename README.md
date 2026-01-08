# GenAI Legal Document Explainer

A comprehensive web application that leverages generative AI to analyze, simplify, and extract insights from legal documents (PDFs).

## Features

- **PDF Processing**: Extracts text from PDFs with fallback OCR for image-based documents
- **Document Classification**: Automatically categorizes clauses into legal domains (employment, rental, etc.)
- **Risk Analysis**: Identifies potential risks and red flags in legal clauses
- **Simplification**: Converts complex legal language into plain English explanations
- **Multi-Query Retrieval**: Semantic search using FAISS vector store for clause lookup
- **Smart Caching**: Reduces API costs by caching classifier and analyzer results
- **Batch Processing**: Efficient batch operations for scaling document analysis

## Project Structure
```text
.
├── app/
│   ├── main.py                 # Streamlit UI
│   ├── config.py               # Configuration & API keys
│   └── __init__.py
│
├── ingestion/
│   ├── loader.py               # PDF text extraction
│   ├── ocr.py                  # Pytesseract OCR handler
│   ├── cleaner.py              # Text cleaning utilities
│   └── __init__.py
│
├── processing/
│   ├── chunker.py              # Document chunking
│   ├── classifier.py           # Single clause classifier
│   ├── batch_classifier.py     # Batch classification with caching
│   ├── simplifier.py           # Clause simplification
│   ├── batch_simplifier.py     # Batch simplification with caching
│   ├── risk_analyzer.py        # Risk assessment
│   ├── batch_risk_analyzer.py  # Batch risk analysis with caching
│   ├── batch_multi_analyzer.py # Combined batch analysis
│   ├── aggregator.py           # Document-level insights
│   ├── answer_synthesizer.py   # Query response generation
│   └── __init__.py
│
├── retrieval/
│   ├── embeddings.py           # HuggingFace embeddings
│   ├── vectorstore.py          # FAISS vector store management
│   ├── retriever.py            # Basic retrieval logic
│   ├── multi_query.py          # Multi-query retrieval with LLM
│   └── __init__.py
│
├── utils/
│   ├── cache_utils.py          # Persistent JSON cache for API results
│   ├── json_utils.py           # JSON parsing utilities
│   └── __init__.py
│
├── data_internal/              # Auto-created cache directory
│   └── cache_store.json        # API result cache
│
├── vectorstore/                # Auto-created FAISS index directory
│
└── requirements.txt
```

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/AnubhavagarwalDev/legal_docs_analyzer_langchain.git
cd legal_docs_analyzer_langchain
```
2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
3. **Setup environment variables:**

Create a `.env` file in the project root and add your Gemini API key:

```env
GEMINI_API_KEY=your_google_gemini_api_key
```
4. **Install Tesseract (for OCR):**
#### Windows
Download and install Tesseract from GitHub:  
👉 https://github.com/UB-Mannheim/tesseract/wiki  
After installation, make sure **Tesseract is added to your PATH**.

---

#### macOS
```bash
brew install tesseract
```
---
#### Linux (Ubuntu / Debian)
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```
---
**(If you use OCR with PDFs, also install Poppler)**
