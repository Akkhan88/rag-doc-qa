import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI


loader = PyPDFLoader("/Users/thomasshelby/Desktop/Formal_Complaint_Form_(TUA_FM_AC_015).pdf")
pages = loader.load()
print(f"loaded {len(pages)} pages")
print(pages[0].page_content[:500])

with open("output.txt", "w") as f:
    f.write(f"loaded {len(pages)} pages\n")
    f.write(pages[0].page_content[:500])

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunks")

embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(chunks, embeddings)
print("Stored in ChromaDB")

llm = ChatOpenAI(model="gpt-4o-mini")

query = input("Ask a question: ")
results = db.similarity_search(query, k=3)

context = "\n".join([r.page_content for r in results])

response = llm.invoke(
    f"Answer the question based ONLY on this context. If the answer is not in the context, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {query}"
)

print(response.content)