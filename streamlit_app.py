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
                    chunks, OpenAIEmbeddings()
                )
                st.session_state.file_key = file_key
                st.success(f"Ready — {len(pages)} pages, {len(chunks)} chunks indexed.")
            finally:
                os.unlink(tmp_path)

    question = st.text_input("Ask a question about the document")

    if st.button("Ask") and question.strip():
        with st.spinner("Searching and generating answer..."):
            db = st.session_state.vectorstore
            hits = db.similarity_search(question, k=3)
            context = "\n\n".join(r.page_content for r in hits)
            llm = ChatOpenAI(model="gpt-4o-mini")
            response = llm.invoke(
                "Answer the question based ONLY on the context below. "
                "If the answer is not in the context, say \"I don't know\".\n\n"
                f"Context:\n{context}\n\nQuestion: {question}"
            )
        st.markdown("**Answer:**")
        st.write(response.content)

        with st.expander("Source chunks used"):
            for i, hit in enumerate(hits, 1):
                st.markdown(f"**Chunk {i}** (page {hit.metadata.get('page', '?')})")
                st.write(hit.page_content)
