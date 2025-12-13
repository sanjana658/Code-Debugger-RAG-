import streamlit as st
import os
import tempfile

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="RAG Code Debugger", layout="wide")
st.title("🧠 RAG Code Debugger")
st.write("Upload code or PDF files and ask questions about them")

uploaded_file = st.file_uploader(
    "Upload a .py, .txt, or .pdf file",
    type=["py", "txt", "pdf"]
)

question = st.text_input("Ask a question")

# -----------------------------
# Main Logic
# -----------------------------
if uploaded_file and question:

    # Save uploaded file
    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Select loader
    if suffix == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # Embeddings (phi3)
    embeddings = OllamaEmbeddings(model="phi3")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Retriever (NEW API)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(question)

    # Build context
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    # LLM (phi3)
    llm = Ollama(model="phi3")

    prompt = f"""
You are a senior software engineer.

Use the following context to answer the question clearly.

Context:
{context}

Question:
{question}
"""

    answer = llm.invoke(prompt)

    st.subheader("🧩 Answer")
    st.write(answer)

    os.remove(file_path)


