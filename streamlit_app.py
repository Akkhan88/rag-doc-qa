import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

load_dotenv()

st.title("RAG Document Q&A")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    file_key = uploaded_file.name + str(uploaded_file.size)

    if st.session_state.get("file_key") != file_key:
        with st.spinner("Processing PDF — embedding chunks..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                pages = PyPDFLoader(tmp_path).load()
                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=50
                ).split_documents(pages)
                st.session_state.vectorstore = Chroma.from_documents(
                    chunks, OpenAIEmbeddings(),
                    persist_directory="./chroma_db"
                )
                st.session_state.file_key = file_key
                st.success(f"Ready — {len(pages)} pages, {len(chunks)} chunks indexed.")
            finally:
                os.unlink(tmp_path)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input("Ask a question about the document")
    llm = ChatOpenAI(model="gpt-4o-mini")

    if question:
        with st.spinner("Searching and generating answer..."):
            db = st.session_state.vectorstore
            hits = db.similarity_search(question, k=3)
            context = "\n\n".join(r.page_content for r in hits)

            history = "\n".join(
                f"User: {q}\nAI: {a}"
                for q, a in st.session_state.chat_history
            )

            response = llm.invoke(
                f"Previous conversation:\n{history}\n\n"
                f"Context:\n{context}\n\n"
                f"Answer based ONLY on the context. "
                f"If not in context, say 'I don't know'.\n\n"
                f"Question: {question}"
            )

            st.session_state.chat_history.append(
                (question, response.content)
            )
            st.write(response.content)
