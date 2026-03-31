# 🦊 GitLab Handbook AI Assistant

An interactive **Retrieval-Augmented Generation (RAG)** chatbot that lets employees and aspiring employees explore GitLab's public [Handbook](https://handbook.gitlab.com/) and [Direction](https://about.gitlab.com/direction/) pages through natural language conversation.

Built with **Streamlit**, **Groq (Llama 3.3 70B)**, **sentence-transformers**, and **FAISS**.

---

## ✨ Features

| Feature | Description |
|---|---|
| **RAG-Powered Answers** | Retrieves relevant handbook sections and generates grounded, accurate responses |
| **Source Citations** | Every answer includes links to the original handbook pages for verification |
| **Streaming Responses** | Answers stream in real-time for a responsive experience |
| **Conversation Memory** | Maintains context across follow-up questions within a session |
| **Guardrails & Safety** | Input validation, PII detection, off-topic filtering, and prompt injection protection |
| **Relevance Scoring** | Transparent relevance scores show how well sources match your question |
| **Suggested Questions** | One-click starter questions to explore the handbook |
| **Local Embeddings** | Embeddings run 100% locally — no API calls, no rate limits for search |
| **Modern UI** | Clean, GitLab-themed interface with responsive design |

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────────┐
│   User       │────▶│  Guardrails  │────▶│  sentence-transformers│
│   Query      │     │  (validate)  │     │  (local embedding)    │
└─────────────┘     └──────────────┘     └──────────┬───────────┘
                                                     │
                                                     ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────────────┐
│  Streaming   │◀────│  Groq /      │◀────│    FAISS             │
│  Response    │     │  Llama 3.3   │     │    Retrieval         │
└─────────────┘     └──────────────┘     └──────────────────────┘
                           │
                    ┌──────┴───────┐
                    │   Source     │
                    │  Citations   │
                    └──────────────┘
```

**Data Flow:**
1. User submits a question → guardrails validate & sanitize input
2. Question is embedded **locally** using sentence-transformers (`all-MiniLM-L6-v2`)
3. FAISS performs similarity search against pre-indexed handbook chunks
4. Top-k relevant chunks + question are sent to **Llama 3.3 70B** via Groq
5. Streamed response is displayed with source citations

**Tech Stack:**
- **Frontend:** Streamlit
- **LLM:** Llama 3.3 70B via Groq (free API)
- **Embeddings:** sentence-transformers `all-MiniLM-L6-v2` (runs locally)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Data Processing:** BeautifulSoup, custom text chunking

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- A Groq API key ([get one free — no credit card](https://console.groq.com/keys))

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/genAI.git
cd genAI
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

**Option A — Environment variable:**
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

**Option B — Streamlit secrets (for deployment):**
```bash
mkdir -p .streamlit
echo 'GROQ_API_KEY = "your-key-here"' > .streamlit/secrets.toml
```

**Option C — Enter it in the UI:**
The sidebar prompts for the key if none is found in the environment.

### 5. Run the chatbot

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. On first launch, it downloads the embedding model (~80MB) and builds the FAISS index locally. This takes ~30 seconds and is cached for instant subsequent starts.

---

## 📁 Project Structure

```
genAI/
├── app.py                      # Streamlit application (UI + orchestration)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore
├── README.md
│
├── .streamlit/
│   └── config.toml             # Streamlit theming (GitLab purple)
│
├── core/
│   ├── __init__.py
│   ├── indexer.py              # FAISS index building with local embeddings
│   ├── retriever.py            # Semantic retrieval + citation extraction
│   ├── chatbot.py              # Groq/Llama 3.3 integration + prompt engineering
│   └── guardrails.py           # Input/output safety guardrails
│
├── utils/
│   ├── __init__.py
│   └── text_processing.py      # Document chunking utilities
│
├── scripts/
│   └── scrape_gitlab.py        # GitLab handbook scraper (refresh data)
│
└── data/
    ├── gitlab_docs.json        # Pre-scraped handbook content (39 pages)
    └── index/                  # Generated FAISS index (auto-created)
```

---

## 🔄 Refreshing the Data

The repository ships with pre-scraped seed data (39 handbook pages). To fetch the latest content:

```bash
python scripts/scrape_gitlab.py
```

This will crawl GitLab's handbook and direction pages and update `data/gitlab_docs.json`. After updating the data, delete the `data/index/` folder so the FAISS index is rebuilt on the next app start:

```bash
rm -rf data/index/
streamlit run app.py
```

---

## ☁️ Deployment

### Streamlit Community Cloud (Recommended)

1. Push the repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo and select `app.py`
4. Add `GROQ_API_KEY` in **Settings → Secrets**:
   ```toml
   GROQ_API_KEY = "your-key-here"
   ```
5. Deploy!

### Hugging Face Spaces

1. Create a new Space with the **Streamlit** SDK
2. Upload all project files
3. Add `GROQ_API_KEY` as a secret in Settings
4. The app will build and deploy automatically

---

## 🛡️ Safety & Guardrails

| Guard | Purpose |
|---|---|
| **Input length limits** | Prevents excessively long queries |
| **PII detection** | Redacts SSNs, credit cards, emails, phone numbers before processing |
| **Prompt injection filter** | Blocks "ignore instructions", "pretend you are" patterns |
| **Off-topic detection** | Identifies queries unrelated to GitLab and provides gentle redirection |
| **Relevance thresholding** | Filters out low-relevance retrieval results |
| **Content safety** | System prompt hardening prevents role abandonment |

---

## 🧠 Key Technical Decisions

| Decision | Rationale |
|---|---|
| **Local embeddings (sentence-transformers)** | Zero API calls for search — no rate limits, no billing, works offline |
| **Groq + Llama 3.3 70B** | Free API, no credit card, extremely fast inference (~500 tokens/sec) |
| **FAISS** | Lightweight, battle-tested vector search, no external services |
| **Chunking with overlap** | Sentence-boundary splitting with 200-char overlap prevents context loss |
| **Streaming responses** | Better UX — users see the answer forming in real-time |
| **Separate retrieval & generation** | Transparency — users see exactly which sources informed the answer |

---

## 🎨 Innovation Highlights

- **Hybrid architecture:** Local embeddings (zero cost, zero latency) + cloud LLM (Groq for speed) — best of both worlds.
- **Transparency-first design:** Relevance scores and source citations are shown prominently, so users can verify answers against the original handbook.
- **Guardrail suite:** Multi-layered input validation protects against PII leaks, prompt injection, and off-topic abuse.
- **Employee-centric UX:** Suggested questions surface key handbook topics; follow-up conversations maintain context for exploratory learning.
- **Modular architecture:** Each component (scraper, indexer, retriever, chatbot, guardrails) is independently testable and replaceable.

---

## 📚 Resources

- [GitLab Handbook](https://handbook.gitlab.com/)
- [GitLab Direction](https://about.gitlab.com/direction/)
- [Groq Console](https://console.groq.com/) (free API key)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FAISS Documentation](https://faiss.ai/)

---

## 📄 License

This project is for educational purposes. GitLab's handbook content is publicly available under their terms of use.
