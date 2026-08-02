import streamlit as st
st.title("Hello World")
st.write("Testing Streamlit")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
st.title("PDF RAG Chatbot")

pdf = st.file_uploader("Upload PDF")

if pdf:

    with open("temp.pdf", "wb") as f:
        f.write(pdf.getbuffer())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(
        chunks,
        embeddings
    )

    query = st.text_input("Ask a question")

    if query:

        docs = db.similarity_search(
            query,
            k=3
        )

        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        llm = ChatOllama(
    model="qwen2.5:3b"
)

        prompt = f"""
        Context:
        {context}

        Question:
        {query}
        """

        response = llm.invoke(prompt)

        st.write(response.content)