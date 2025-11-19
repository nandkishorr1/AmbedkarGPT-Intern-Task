# AmbedkarGPT — Phase 1 (RAG Command-Line Prototype)

This project is a simple RAG-based question answering system built using:
- LangChain
- ChromaDB
- HuggingFace embeddings (all-MiniLM-L6-v2)
- Ollama (Mistral 7B)

The system loads the provided speech.txt, builds a vector store, and answers questions entirely offline.

---

## Setup Instructions

### 1. Create virtual env
Windows:
    python -m venv venv
    venv\Scripts\activate

Linux/Mac:
    python -m venv venv
    source venv/bin/activate

### 2. Install dependencies
    pip install -r requirements.txt

### 3. Install Ollama and pull model
    curl -fsSL https://ollama.ai/install.sh | sh
    ollama pull mistral

### 4. Run the program
    python main.py

You will see a prompt. Ask questions about the speech, e.g.:
- What is the remedy against caste?
- Why does Ambedkar oppose the shastras?

Type 'exit' to quit.

---

## Notes
- Chroma vector DB is saved in chroma_db/ automatically.
- Everything works offline.
