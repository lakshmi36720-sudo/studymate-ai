# StudyMate AI 🎓

An intelligent, modular **AI Study Assistant** powered by **Retrieval-Augmented Generation (RAG)** and **FAISS (Facebook AI Similarity Search)** vector indexing.

---

## 🌟 Features

- 📄 **Multi-Format Document Ingestion**: Supports PDF, TXT, and Markdown (`.md`) study notes and textbooks.
- ⚡ **Fast FAISS Vector Search**: Vectorizes document chunks for fast semantic similarity retrieval.
- 🧠 **Context-Aware Q&A**: Answers questions strictly based on uploaded course material with source page citations.
- 📝 **Automatic Study Summaries**: Generates high-level summaries and key bullet point takeaways.
- 🎯 **Practice Quiz Generator**: Creates 5-question quizzes with answer keys to test retention.
- 💻 **Dual Interfaces**: Choose between an interactive **Streamlit Web UI** or a lightweight **CLI Tool**.

---

## 📁 Project Structure

```
studymate-ai/
├── .env.example           # Template for environment variables and API keys
├── .gitignore             # Standard gitignore (ignores virtual environments & vector indexes)
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── app.py                 # Streamlit Web Application entry point
├── main.py                # Command Line Interface (CLI) entry point
├── data/
│   ├── docs/              # Directory for user uploaded study documents (PDF, TXT, MD)
│   └── vectorstore/       # Directory where FAISS indices are persisted
└── src/
    ├── __init__.py        # Package initialization
    ├── config.py          # Centralized configuration & environment loader
    ├── document_loader.py # Document parsing & text chunking logic
    ├── embeddings.py      # HuggingFace & OpenAI embedding models loader
    ├── vector_store.py    # FAISS vector store creation, loading & querying
    └── rag_chain.py       # RAG pipeline for Q&A, Summarization & Quiz generation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ installed
- Git installed

### 2. Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lakshmi36720-sudo/studymate-ai.git
   cd studymate-ai
   ```

2. **Create and activate a virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Environment Variables:**
   Copy `.env.example` to `.env` and fill in your keys (e.g. OpenAI API key if using OpenAI models):
   ```bash
   cp .env.example .env
   ```

---

## 💻 Usage

### Option A: Launch Web Interface (Streamlit)
```bash
streamlit run app.py
```
- Open your browser at `http://localhost:8501`
- Upload your course materials in the sidebar and click **Build / Rebuild Vector Index**
- Chat with your documents, generate summaries, or take practice quizzes!

### Option B: Command Line Interface (CLI)

1. Place your study documents in `data/docs/`
2. Index your documents into FAISS:
   ```bash
   python main.py --index
   ```
3. Start the interactive terminal Q&A:
   ```bash
   python main.py
   ```

---

## 🛠️ Architecture & RAG Workflow

```
[ Study Notes / PDFs ] ──> [ Document Loader & Splitter ] ──> [ Embedding Model ]
                                                                      │
                                                                      ▼
[ User Question ] ──> [ Similarity Search ] <────────────── [ FAISS Vector Store ]
         │                        │
         └───────────┬────────────┘
                     ▼
             [ Context + Prompt ] ──> [ LLM Response Generator ] ──> [ Answer + Citations ]
```

---

## 📄 License
MIT License. Feel free to modify and adapt StudyMate AI for your personal study needs!
