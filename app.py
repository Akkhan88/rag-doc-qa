import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI


pdf_paths = [
    "/Users/thomasshelby/Desktop/PST107/PST107_Week5_Random_variables_distributions.pdf",
    "/Users/thomasshelby/Desktop/PST107/PST107_Module6_Expectation.pdf",
    "/Users/thomasshelby/Desktop/PST107/PST107_Week7_Special_Distributions.pdf",
    "/Users/thomasshelby/Desktop/PST107/PST107_Week8_Large_Random_Samples.pdf",
]

def load_pdf(path):
    print(f"Loading: {os.path.basename(path)}")
    result = PyPDFLoader(path).load()
    print(f"Done: {os.path.basename(path)} ({len(result)} pages)")
    return result

with ThreadPoolExecutor() as executor:
    results = executor.map(load_pdf, pdf_paths)

pages = [page for doc_pages in results for page in doc_pages]
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

while True:
    query = input("\nAsk a question (type 'quit' to exit): ")
    if query.lower() == 'quit':
        break
    results = db.similarity_search(query, k=3)
    context = "\n".join([r.page_content for r in results])
    response = llm.invoke(
        f"Answer the question based ONLY on this context. If the answer is not in the context, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {query}"
    )
    print(response.content)