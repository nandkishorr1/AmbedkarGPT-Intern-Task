import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# 1. Load the speech
loader = TextLoader("speech.txt")
docs = loader.load()

# 2. Split into chunks
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Create vector DB
vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="db")
vectordb.persist()

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 5. LLM (Mistral via Ollama)
llm = Ollama(model="mistral")

def generate_answer(question):
    # Retrieve context
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are an assistant answering only from the provided context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = llm.invoke(prompt)
    return response

print("AmbedkarGPT RAG System Ready!")
while True:
    q = input("\nAsk a question (or type 'exit'): ")
    if q.lower() == "exit":
        break

    answer = generate_answer(q)
    print("\nAnswer:", answer)
