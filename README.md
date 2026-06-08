# Policy & Process Copilot — Agentic RAG Assistant

A production-quality AI assistant that answers questions over company policy documents using **Retrieval-Augmented Generation (RAG)** and an **Agentic workflow** built with LangChain, LangGraph, ChromaDB, OpenAI, and Streamlit.

---

## Table of Contents
1. [What This Project Does](#1-what-this-project-does)
2. [Architecture Explanation](#2-architecture-explanation)
3. [How Agentic RAG Works](#3-how-agentic-rag-works)
4. [Project Structure](#4-project-structure)
5. [Setup Steps](#5-setup-steps)
6. [Running the Application](#6-running-the-application)
7. [Using the Application](#7-using-the-application)
8. [Technology Stack](#8-technology-stack)

---

## 1. What This Project Does

This assistant can:
- **Answer questions** about company HR policies, IT security, benefits, onboarding, etc.
- **Cite sources** for every answer (document name + chunk reference)
- **Decide intelligently** whether to search documents, ask a clarifying question, or answer directly
- **Verify** its own retrieval results before generating an answer
- **Retry** with a rewritten query if the first retrieval attempt fails to find useful content

---

## 2. Architecture Explanation

```
┌─────────────────────────────────────────────────────┐
│                 STREAMLIT WEB UI                    │
│   (Chat interface + Agent trace + Citations)        │
└──────────────────────┬──────────────────────────────┘
                       │ user query
                       ▼
┌─────────────────────────────────────────────────────┐
│             AGENTIC RAG WORKFLOW (LangGraph)        │
│  Router → Retriever → Verifier → Answer Generator  │
└──────────────────────┬──────────────────────────────┘
                       │ semantic search
                       ▼
┌─────────────────────────────────────────────────────┐
│        VECTOR DATABASE (ChromaDB — local)           │
│   Stores embeddings of all policy document chunks   │
└──────────────────────┬──────────────────────────────┘
                       │ similarity search
                       ▼
┌─────────────────────────────────────────────────────┐
│         DOCUMENT STORE  (documents/ folder)         │
│   TXT, PDF, Markdown policy files (10–25 docs)     │
└─────────────────────────────────────────────────────┘
```

**Key Components:**

| Component | File | What It Does |
|-----------|------|-------------|
| Document Loader | `rag/document_loader.py` | Reads PDF/TXT/MD files, splits into 1000-char chunks with 200-char overlap |
| Vector Store | `rag/vector_store.py` | Embeds chunks with OpenAI, stores in ChromaDB, searches by semantic similarity |
| Agents | `rag/agents.py` | LangGraph workflow with Router, Retriever, Verifier, Query Rewriter, Answer Generator |
| UI | `app.py` | Streamlit web app with chat interface, agent trace, and citations |

---

## 3. How Agentic RAG Works

### Step-by-Step Flow

```
User asks: "How many days of annual leave do I get?"
                │
         ┌──────▼──────┐
         │   ROUTER    │  ← Analyses query, decides: RETRIEVE
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │  RETRIEVER  │  ← Searches ChromaDB, finds top 6 relevant chunks
         └──────┬──────┘    from leave_policy.txt
                │
         ┌──────▼──────┐
         │  VERIFIER   │  ← Checks: "Do these chunks contain enough info?"
         └──────┬──────┘
                │
          Sufficient?  ──── NO ──► QUERY REWRITER → RETRIEVER (retry once)
                │
               YES
                │
         ┌──────▼──────┐
         │   ANSWER    │  ← Generates answer with [Source N] citations
         └──────┬──────┘
                │
       "Employees are entitled to 21 working days of annual leave [Source 1]..."
```

### The Three Actions the Router Can Take

| Action | When Used | Example |
|--------|-----------|---------|
| `RETRIEVE` | Question is about company policy | "How do I claim expenses?" |
| `CLARIFY` | Question is too vague | "Tell me about it" |
| `DIRECT_ANSWER` | General knowledge / greeting | "Hello!" |

### Why Agentic?
A simple RAG would always search documents — even for greetings. This wastes API calls and gives strange answers. The Router Agent *decides* what to do. The Verifier Agent *checks* whether the retrieval was successful before generating an answer. This makes the system more reliable and accurate.

---

## 4. Project Structure

```
PolicyRAG/
│
├── app.py                        ← Streamlit web UI (run this)
├── requirements.txt              ← All Python dependencies
├── .env.example                  ← API key template (copy to .env)
├── README.md                     ← This file
├── test_dataset.json             ← 10 test questions with expected answers
│
├── rag/
│   ├── __init__.py
│   ├── document_loader.py        ← Load & chunk PDF/TXT/MD files
│   ├── vector_store.py           ← ChromaDB + OpenAI embeddings
│   └── agents.py                 ← LangGraph agentic workflow
│
├── documents/                    ← Put your policy files here
│   ├── leave_policy.txt
│   ├── expense_reimbursement_policy.txt
│   ├── incident_response_playbook.txt
│   ├── remote_work_policy.txt
│   ├── performance_review_process.txt
│   ├── benefits_guide.txt
│   ├── onboarding_guide.txt
│   ├── it_security_policy.txt
│   ├── training_and_development.txt
│   ├── code_of_conduct.txt
│   ├── hr_general_policies.txt
│   └── data_privacy_policy.txt
│
├── chroma_db/                    ← Auto-created: ChromaDB persistent storage
└── logs/                         ← Auto-created: daily log files
```

---

## 5. Setup Steps

### Prerequisites
- Python 3.10 or higher
- An OpenAI API key (get one at https://platform.openai.com/api-keys)

### Step 1 — Clone or copy the project
Make sure you are in the `PolicyRAG` directory.

### Step 2 — Create a virtual environment
A virtual environment is an isolated Python installation so this project's packages don't conflict with your system.

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

After activation, your terminal prompt will show `(venv)`.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs LangChain, LangGraph, ChromaDB, Streamlit, OpenAI, and all other required packages. It may take 2–3 minutes.

### Step 4 — Set up your API key

Copy the example file and add your key:

```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

Now open `.env` in any text editor and replace `sk-your-openai-api-key-here` with your actual OpenAI API key:

```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Never share your `.env` file or commit it to Git.**

### Step 5 — Verify documents are in place

The `documents/` folder already contains 12 sample policy files. You can add your own `.txt`, `.pdf`, or `.md` files at any time.

---

## 6. Running the Application

With the virtual environment active and `.env` configured:

```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`.

**First run takes ~10 seconds** while Streamlit and the AI models load.

---

## 7. Using the Application

### Initial Setup (One-Time)
1. In the **sidebar**, confirm your API key is shown as loaded.
2. The `documents` folder path is pre-filled. Click **➕ Ingest Docs**.
3. Wait for the ingestion to complete (progress shown in sidebar). You'll see the chunk count update.
4. Done! You can now ask questions.

### Asking Questions
- Type your question in the chat box at the bottom and press Enter.
- The assistant will show:
  - **The answer** with inline `[Source N]` citations
  - **A coloured badge** showing the action taken (RETRIEVE / CLARIFY / DIRECT_ANSWER)
  - **Sources used** — the specific documents referenced
  - **Agent reasoning trace** (click "🔍 Agent reasoning trace" to expand)

### Adding Your Own Documents
1. Copy your `.txt`, `.pdf`, or `.md` files into the `documents/` folder.
2. Click **🗑️ Clear DB** in the sidebar to remove old embeddings.
3. Click **➕ Ingest Docs** to re-ingest everything.

### Switching Models
Use the **Model Settings** dropdown in the sidebar:
- `gpt-4o` — Best accuracy (recommended)
- `gpt-4o-mini` — Faster and 10x cheaper, slightly less accurate
- `gpt-4-turbo` — Older, high-quality model

---

## 8. Technology Stack

| Technology | Version | Role |
|-----------|---------|------|
| Python | 3.10+ | Programming language |
| LangChain | 0.3+ | Document loading, text splitting, embeddings, LLM abstraction |
| LangGraph | 0.2+ | Agentic workflow graph (Router → Retriever → Verifier → Answer) |
| ChromaDB | 0.5+ | Local vector database for semantic search |
| OpenAI | 1.50+ | LLM (gpt-4o) for reasoning and generation; text-embedding-3-small for embeddings |
| Streamlit | 1.39+ | Web UI framework |
| python-dotenv | 1.0+ | Loads API keys from .env file |
| pypdf | 4.0+ | PDF text extraction |

---

*Built as part of the GenAI Agentic RAG Starter Project.*
