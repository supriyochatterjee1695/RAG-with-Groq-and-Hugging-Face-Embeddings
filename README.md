# CSeametry RAG Engine

The **CSeametry RAG Engine** is a high-performance AI system that transforms static PDF documents into dynamic, searchable knowledge bases.

Unlike basic RAG pipelines, this engine uses a **decoupled architecture**, separating document ingestion from the chat interface—resulting in faster performance, better scalability, and cleaner deployments.

By combining **Groq’s ultra-fast inference** with **local embeddings**, the system delivers near-instant responses with:
- Zero cloud embedding cost  
- 100% data privacy  
- Source-grounded answers with citations  

---

## Key Features

- Lightning-fast responses using Groq LPU inference  
- Fully local embedding pipeline (no external embedding API calls)  
- FAISS-powered semantic search  
- Source citations with page numbers and snippets  
- Modular architecture for scalability  
- Cost-efficient and privacy-first design  

---

## System Architecture

The engine is divided into two independent modules:

### ingest.py (The Builder)
- Loads raw PDFs and optional metadata  
- Splits text using Recursive Character Splitting to preserve semantic meaning  
- Generates dense vector embeddings  
- Stores vectors in a local FAISS index  

### main.py (The Brain)
- Loads the pre-built FAISS index instantly  
- Handles retrieval + generation workflow  
- Uses Groq for ultra-fast LLM responses  
- Returns answers with source references (page numbers + snippets)  

---

## Tech Stack

- Inference Engine: Groq Cloud (llama-3.3-70b-versatile)  
- Embeddings: BAAI/bge-small-en-v1.5 (local CPU execution)  
- Vector Store: FAISS (Facebook AI Similarity Search)  
- Orchestration: LangChain  
- Environment Management: python-dotenv  

---

## Getting Started

### 1. Prerequisites

- Python 3.9 or higher  
- Groq API Key → https://console.groq.com  

---

### 2. Installation

```bash
git clone https://github.com/yourusername/cseametry-rag-engine.git
cd cseametry-rag-engine

pip install groq faiss-cpu sentence-transformers langchain-huggingface langchain-groq python-dotenv pypdf
```

---

### 3. Configuration

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

---

## Usage

### Step 1: Ingest Your Data

Place your PDF files (e.g., `CSeametry.pdf`) in the root directory and run:

```bash
python ingest.py
```

This will generate a `faiss_index/` folder containing your vector database.

---

### Step 2: Chat with Your Documents

Start the interactive assistant:

```bash
python main.py
```

You can now query your documents in natural language and receive context-aware answers with citations.

---

## Example Use Cases

- Executive Search: Query annual reports or financial documents instantly  
- Technical Support: Turn product manuals into an AI helpdesk  
- Legal Review: Extract clauses and identify risks from contracts  
- Training & Onboarding: Convert internal documents into an interactive mentor  

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

Supriyo – CSeametry
Email: supriyochatterjee@cseametry.co.in
Website: https://cseametry.co.in

---

Built with ❤️ by **CSeametry** — Bridging the gap between data and growth.
