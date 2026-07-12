# Context-Aware Chatbot Using LangChain + RAG

A conversational chatbot that retrieves answers from a vectorized knowledge base
(Wikipedia articles on AI/ML topics) and remembers the conversation as it goes,
built with LangChain, FAISS, Groq (Llama 3.3 70B), and Streamlit.

## How it works

1. **Ingestion** (`ingest.py`, run once): downloads a set of Wikipedia articles,
   splits them into overlapping chunks, embeds each chunk into a vector using a
   free local HuggingFace model, and saves the vectors into a FAISS index.
2. **Chat** (`app.py`): when you ask a question, it gets embedded and matched
   against the FAISS index to retrieve the most relevant chunks. Those chunks,
   plus the running conversation history, are sent to the LLM (Groq/Llama 3.3)
   to generate a grounded answer.
3. **Memory**: `ConversationBufferMemory` keeps track of previous turns, so
   follow-up questions like "what about its limitations?" resolve correctly
   without you having to repeat context.

## Project structure

```
rag-chatbot/
├── ingest.py           # Builds the vector store (run once)
├── app.py               # Streamlit chatbot app
├── requirements.txt
├── vectorstore/          # Created by ingest.py (FAISS index)
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

If you're on Windows and hit wheel-build errors on `faiss-cpu` or
`sentence-transformers` (common on Python 3.13), the fix is the same one that
worked for the BERT classifier project: use Python 3.10–3.12 instead, or run
this on Google Colab where all wheels are prebuilt.

### 2. Get a free Groq API key

Go to [console.groq.com](https://console.groq.com) → sign up → create an API
key. No credit card required. Llama 3.3 70B is available on the free tier.

### 3. Build the knowledge base

```bash
python ingest.py
```

This downloads the Wikipedia articles listed in `TOPICS` (edit that list to
change the corpus), chunks them, embeds them, and saves the result to
`./vectorstore/`. You only need to run this once — re-run it if you change the
topics or add your own documents.

**Why this step needs internet access**: it calls the live Wikipedia API. If
you're running this in a sandboxed/offline environment, it will fail — that's
expected there, not a bug. It works normally on your own machine or in Colab.

### 4. Run the chatbot

```bash
streamlit run app.py
```

Paste your Groq API key into the sidebar and start chatting.

## Using your own documents instead of Wikipedia

Open `ingest.py` and replace the `load_documents()` function. For a folder of
text files:

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
loader = DirectoryLoader("data/", glob="*.txt", loader_cls=TextLoader)
documents = loader.load()
```

For PDFs:

```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("my_document.pdf")
documents = loader.load()
```

Everything downstream (chunking, embedding, saving) stays the same.

## Why things are pinned the way they are

`requirements.txt` pins `langchain==0.3.25` deliberately. LangChain 1.x
(released later) removed the classic `ConversationalRetrievalChain` and
`ConversationBufferMemory` classes this project uses, in favor of a new
runnable/LangGraph-based memory system. The 0.3.x line is simpler for a
course project and is still what most tutorials and documentation you'll
find are written against — but it's worth knowing this project intentionally
avoids the newest LangChain if you go looking for docs later.

## Deploying

Push this repo to GitHub, then deploy for free on
[Streamlit Community Cloud](https://streamlit.io/cloud):
1. Connect your GitHub repo
2. Set the main file to `app.py`
3. Add your `GROQ_API_KEY` under **Settings → Secrets** as:
   ```
   GROQ_API_KEY = "your-key-here"
   ```
4. You'll also need to run `ingest.py` once and commit the resulting
   `vectorstore/` folder to the repo (or add a build step that runs it),
   since Streamlit Cloud won't run `ingest.py` for you automatically.

## Troubleshooting

| Problem | Why it happens | Fix |
|---|---|---|
| `No vector store found` | You ran `app.py` before `ingest.py` | Run `python ingest.py` first |
| `403`/connection error during `ingest.py` | No internet access, or Wikipedia rate-limited you | Retry, or reduce the `TOPICS` list |
| Slow first run of `app.py` | The embedding model (~90MB) downloads and caches on first use | Only happens once; subsequent runs are fast |
| `ImportError` on `ConversationBufferMemory` | You have LangChain 1.x installed instead of 0.3.x | `pip install -r requirements.txt` again to re-pin versions |
| Wheel build errors on Windows | Some packages (faiss, sentence-transformers) lack prebuilt wheels for very new Python versions | Use Python 3.10–3.12, or develop on Colab |

## Skills demonstrated

- Conversational AI development (LangChain conversational chain + memory)
- Document embedding and vector search (FAISS + sentence-transformers)
- Retrieval-Augmented Generation (RAG)
- LLM integration and deployment (Groq API + Streamlit)
