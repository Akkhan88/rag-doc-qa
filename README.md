RAG Document Q&A System
A Retrieval-Augmented Generation (RAG) system that lets you ask questions about any document and get instant, accurate answers powered by AI.
What It Does
Upload any document (PDF, Word, CSV, TXT) → the system reads it, understands it, and answers your questions based only on the document's content. If the answer isn't in the document, it tells you i don't know instead of making things up.
How It Works
Document → Load → Chunk → Embed → Store in ChromaDB → User asks question → Find relevant chunks → AI answers

Load — Reads the document and extracts all text
Chunk — Splits text into small pieces (~500 characters) so only relevant parts are searched
Embed — Converts each chunk into numerical representations (embeddings) that capture meaning
Store — Saves embeddings in ChromaDB vector database for fast similarity search
Query — Converts the user's question into embeddings, finds the 3 closest chunks, sends them to the AI model to generate an accurate answer

Tech Stack

Python
LangChain — Framework for connecting LLMs to data
OpenAI GPT-4o-mini — Language model for generating answers
OpenAI Embeddings — Converts text to meaning-based numerical vectors
ChromaDB — Vector database for storing and searching embeddings
PyPDF / Docx2txt — Document loaders for multiple file formats

Supported File Types
FormatLoaderPDFPyPDFLoaderWord (.docx)Docx2txtLoaderCSVCSVLoaderText (.txt)TextLoader
Setup
bashgit clone https://github.com/Akkhan88/rag-doc-qa.git
cd rag-doc-qa
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Create a .env file:
OPENAI_API_KEY=your-api-key-here
Usage
Drop any document into the project folder, update the filename in app.py, then run:
bashpython app.py
Ask questions in plain English. Type quit to exit.
Example
Ask a question: How do I file a formal complaint?
> To file a formal complaint, you need to complete form TUA FM AC-015...

Ask a question: What is the deadline?
> Based on the document, complaints must be submitted within 20 business days...

Ask a question: What is the capital of France?
> I don't know — this information is not in the provided document.
Architecture
┌──────────┐     ┌──────────┐     ┌───────────────┐     ┌──────────┐
│ Document │ ──▶ │ Chunker  │ ──▶ │ Embeddings    │ ──▶ │ ChromaDB │
│ (PDF)    │     │ (500char)│     │ (OpenAI)      │     │ (Vector) │
└──────────┘     └──────────┘     └───────────────┘     └────┬─────┘
                                                             │
┌──────────┐     ┌──────────┐     ┌───────────────┐         │
│  Answer  │ ◀── │ GPT-4o   │ ◀── │ Top 3 Chunks  │ ◀───────┘
│          │     │ mini     │     │ (by meaning)  │
└──────────┘     └──────────┘     └───────────────┘
Built By
Ammar — AI Engineering Student | Building production AI systems

Part of my AI Engineer portfolio. Built from scratch as Project #1.
