# AmbedkarGPT-Intern-Task

A command-line Q&A system that answers questions from a short excerpt of Dr. B.R. Ambedkar's "Annihilation of Caste".  
Tech stack: Python 3.8+, LangChain, ChromaDB, HuggingFace (MiniLM), Ollama (Mistral 7B).

## Setup Instructions

1. **Clone this repo:**
   ```
   git clone https://github.com/<your_handle>/AmbedkarGPT-Intern-Task.git
   cd AmbedkarGPT-Intern-Task
   ```

2. **Set up a Python environment and install dependencies:**
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Install and run Ollama (for local Mistral 7B):**
   - Download Ollama:  
     `curl -fsSL https://ollama.ai/install.sh | sh`
   - Pull the Mistral model:  
     `ollama pull mistral`
   - Start Ollama service:
     ```
     ollama serve
     ```

4. **Run AmbedkarGPT:**
   ```
   python main.py
   ```

5. **Ask questions!**  
   Try things like:
   - What does Dr. Ambedkar say about the shastras?
   - Why is caste hard to abolish?
   - What analogy does he use for social reform?
   - Type `exit` to quit.

## File Descriptions

- `main.py` — Main code orchestrating the RAG pipeline.
- `requirements.txt` — Python dependencies.
- `speech.txt` — The short excerpt of Dr. Ambedkar's speech.
- `README.md` — This file.

## Notes

- No API keys, no online accounts needed.
- Everything runs locally using free, open-source tools.
- To reset or rebuild the vector DB, simply delete the `db/` directory.

---