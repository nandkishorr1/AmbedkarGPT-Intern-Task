"""
AmbedkarGPT-Intern-Task
A simple command-line Q&A system powered by LangChain, ChromaDB, HuggingFaceEmbeddings, and Ollama (Mistral 7B).
- Loads Dr. B.R. Ambedkar's short speech from 'speech.txt'.
- Builds a local vector store and answers questions using RAG (Retrieval Augmented Generation).
"""
from langchain.document_loaders import TextLoader
from langchain.text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

def main():
    # 1. Load the provided text file
    loader = TextLoader("speech.txt")
    documents = loader.load()

    # 2. Split the text into manageable chunks
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # 3. Create Embeddings and store them in a local vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory="db")
    vectordb.persist()

    # 4. Configure the LLM (Ollama, using Mistral 7B)
    llm = Ollama(model="mistral")

    # 5. Build the Retrieval-QA Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",          # "stuff" for simple QA (default)
        retriever=vectordb.as_retriever(search_kwargs={"k": 2})  # Retrieve top 2 chunks
    )

    print("Welcome to AmbedkarGPT! Ask questions about the provided speech. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        response = qa.run(user_input)
        print("Answer:", response, "\n")

if __name__ == "__main__":
    main()