import streamlit as st
import os
import tempfile

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="RAG Code Debugger", layout="wide")
st.title("🧠 RAG Code Debugger")
st.markdown("""
Upload code or PDF files and ask questions about them.  
The AI will analyze your code, identify potential bugs, explain errors, and suggest improvements.
""")

# -----------------------------
# Sidebar Instructions
# -----------------------------
with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. Upload a Python (`.py`), text (`.txt`), or PDF (`.pdf`) file.  
    2. Ask a question about your code.  
    3. View the AI's answer below.  
    4. Optionally, download the answer.
    """)
    st.write("💡 Example questions:")
    st.write("""
    - Why does this code crash for some inputs?  
    - Explain this function step by step.  
    - How can I fix the division by zero error?  
    """)

# -----------------------------
# Columns Layout
# -----------------------------
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader(
        "Upload your code or PDF",
        type=["py", "txt", "pdf"]
    )
    question = st.text_input("Ask a question")

with col2:
    if uploaded_file and question:

        # Save uploaded file temporarily
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        # Select appropriate loader
        if suffix == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        documents = loader.load()

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(documents)

        # Create embeddings (phi3)
        embeddings = OllamaEmbeddings(model="phi3")

        # Vector DB
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        # Retriever
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(question)

        # Build context
        context = "\n\n".join(doc.page_content for doc in relevant_docs)

        # Load LLM
        llm = Ollama(model="phi3")

        prompt = f"""
You are a senior software engineer.

Use the following context to answer the question clearly, and provide fixes if necessary.

Context:
{context}

Question:
{question}
"""
        answer = llm.invoke(prompt)

        # Display answer
        st.subheader("🧩 Answer")
        st.write(answer)

        # Optional: show retrieved context
        with st.expander("Show retrieved context"):
            for i, doc in enumerate(relevant_docs, start=1):
                st.markdown(f"**Chunk {i}:**")
                st.code(doc.page_content, language='python')

        # Download answer
        st.download_button(
            label="Download Answer",
            data=answer,
            file_name="rag_debugger_answer.txt"
        )

        # Cleanup temporary file
        os.remove(file_path)

    else:
        st.info("Upload a file and type your question to get started.")
