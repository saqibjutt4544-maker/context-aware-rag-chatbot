"""
ingest.py
Builds the vector knowledge base for the RAG chatbot.

What this does:
1. Downloads a set of Wikipedia articles (the "custom corpus").
2. Splits them into overlapping chunks (so context isn't cut mid-idea).
3. Embeds every chunk into a vector using a free local HuggingFace model.
4. Saves the vectors into a FAISS index on disk (vectorstore/).

Run this ONCE before starting the chatbot (app.py).
Re-run it any time you change TOPICS or add your own documents.

    python ingest.py
"""

import os
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------
# 1. CONFIGURE YOUR CORPUS
# ---------------------------------------------------------------------------
# Add/remove topics here. Each topic pulls the full Wikipedia article.
# You can swap this whole section out for your own PDFs/text files later
# (see the "Using your own documents instead" note at the bottom).
TOPICS = [
    # Core ML/DL concepts
    "Machine learning",
    "Deep learning",
    "Natural language processing",
    "Transformer (deep learning architecture)",
    "Artificial neural network",
    "Convolutional neural network",
    "Reinforcement learning",
    "Large language model",

    # Machine learning algorithms
    "Linear regression",
    "Logistic regression",
    "Decision tree learning",
    "Random forest",
    "Support vector machine",
    "K-nearest neighbors algorithm",
    "Naive Bayes classifier",
    "K-means clustering",
    "Gradient boosting",
    "Gradient descent",
]

VECTORSTORE_DIR = "vectorstore"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    """Fetch each topic's Wikipedia page as a LangChain Document."""
    all_docs = []
    for topic in TOPICS:
        print(f"Fetching: {topic}")
        try:
            # load_max_docs=1 -> just the best-matching page per topic
            docs = WikipediaLoader(query=topic, load_max_docs=1).load()
            all_docs.extend(docs)
            print(f"  -> got {len(docs)} document(s)")
        except Exception as e:
            # Common cause: no internet access, or Wikipedia rate limiting.
            print(f"  -> FAILED for '{topic}': {e}")
    return all_docs


def build_vectorstore(documents):
    """Chunk documents, embed them, and save a FAISS index to disk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"\nSplit {len(documents)} documents into {len(chunks)} chunks.")

    print(f"Loading embedding model ({EMBEDDING_MODEL})... this downloads once and is cached.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Embedding chunks and building FAISS index (this may take a minute)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"\nSaved vector store to ./{VECTORSTORE_DIR}/")


if __name__ == "__main__":
    documents = load_documents()

    if not documents:
        raise SystemExit(
            "No documents were loaded. Check your internet connection, "
            "or switch to the 'Using your own documents' approach below."
        )

    build_vectorstore(documents)
    print("\nDone. You can now run: streamlit run app.py")

# ---------------------------------------------------------------------------
# Using your own documents instead of Wikipedia
# ---------------------------------------------------------------------------
# Replace load_documents() with something like:
#
#   from langchain_community.document_loaders import DirectoryLoader, TextLoader
#   loader = DirectoryLoader("data/", glob="*.txt", loader_cls=TextLoader)
#   documents = loader.load()
#
# or for PDFs:
#   from langchain_community.document_loaders import PyPDFLoader
#   loader = PyPDFLoader("my_document.pdf")
#   documents = loader.load()
#
# Everything else (chunking, embedding, saving) stays exactly the same.