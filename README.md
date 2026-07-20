---
title: Context-Aware RAG Chatbot
emoji: 💬
colorFrom: blue
colorTo: purple
sdk: streamlit
tags:
  - langchain
  - rag
  - chatbot
  - streamlit
  - faiss
  - groq
  - llm
  - nlp
pinned: false
short_description: Conversational chatbot with context memory and RAG-based retrieval from a vectorized Wikipedia knowledge base
license: mit
---

# 💬 Context-Aware RAG Chatbot

A conversational chatbot that **remembers your conversation** and **retrieves real facts** from a knowledge base — instead of guessing.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview

A Retrieval-Augmented Generation (RAG) chatbot that combines a vectorized knowledge base
with persistent conversation memory. Documents (Wikipedia articles on ML/AI concepts and
algorithms) are chunked and embedded into a **FAISS** vector store. Each user question is
matched against that store to retrieve the most relevant passages, which are combined with
the ongoing chat history and sent to **Llama 3.3 70B** (via the free Groq API) to generate
a grounded answer — all orchestrated with **LangChain** and served through a **Streamlit** UI.

**Core idea:** an LLM answering from retrieved, real documents is far more trustworthy than
one answering purely from memory — if the answer isn't in the knowledge base, the bot says
so instead of hallucinating.

## ✨ Features

- 🧠 **Conversational Memory** — follow-up questions like "what about its limitations?" resolve correctly using chat history
- 📚 **Vectorized Knowledge Base** — Wikipedia articles on ML/AI concepts and algorithms, chunked and embedded with `sentence-transformers`
- 🔍 **Semantic Retrieval** — FAISS similarity search pulls the most relevant chunks for every query
- 🚫 **Grounded Answers** — the bot says "I don't know" when the knowledge base doesn't cover a topic, instead of making things up
- 📎 **Source Transparency** — every answer shows an expandable panel with the exact passages used to generate it
- ⚡ **Fast Inference** — Llama 3.3 70B served through Groq's LPU hardware, free tier, no credit card required
- 🔌 **Swappable Corpus** — easily point the ingestion script at your own PDFs or text files instead of Wikipedia

## 🖥️ Running Locally

**1. Clone the repository**
```bash
git clone https://github.com/saqibjutt4544-maker/context-aware-rag-chatbot.git
cd context-aware-rag-chatbot
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Get a free Groq API key**
Sign up at [console.groq.com](https://console.groq.com) → create an API key. No credit card required.

**5. Build the knowledge base**
```bash
python ingest.py
```
Downloads the Wikipedia articles listed in `TOPICS`, chunks them, embeds them, and saves
a FAISS index to `vectorstore/`. Run this once — re-run it any time you edit `TOPICS`.

**6. Run the chatbot**
```bash
streamlit run app.py
```
Paste your Groq API key into the sidebar and start chatting.

## 🔍 How It Works

```
Documents (Wikipedia)                     User Question
        │                                       │
        ▼                                       ▼
  Chunk + Embed (MiniLM)                  Embed Query
        │                                       │
        ▼                                       ▼
  FAISS Vector Store  ─────────────────►  Similarity Search
                                                 │
                                                 ▼
                                        Retrieved Context
                                                 │
                              Chat Memory ──────►│
                                                 ▼
                                        LangChain Chain
                                                 │
                                                 ▼
                                   LLM (Groq / Llama 3.3 70B)
                                                 │
                                                 ▼
                                         Grounded Answer
```

## 🗂️ Project Structure

```
📦 context-aware-rag-chatbot
┣ 🐍 ingest.py              ← builds the FAISS vector store from Wikipedia
┣ 🐍 app.py                 ← Streamlit chatbot UI with retrieval + memory
┣ 📁 vectorstore/           ← FAISS index (created by ingest.py)
┣ 📄 requirements.txt
┣ 📄 .gitignore
┣ 📄 LICENSE
┗ 📄 README.md
```

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| LangChain | RAG orchestration, conversational chain, memory |
| FAISS | Local vector similarity search |
| sentence-transformers (MiniLM) | Free local document embeddings |
| Groq API (Llama 3.3 70B) | LLM inference, free tier, no credit card |
| Streamlit | Chat UI and deployment |

## 🗺️ Roadmap

- [x] Build vectorized knowledge base from Wikipedia
- [x] Implement retrieval-augmented question answering
- [x] Add conversational memory across turns
- [x] Expand knowledge base with core ML algorithms
- [x] Display retrieved sources per answer
- [ ] Support user-uploaded PDFs from the Streamlit sidebar
- [ ] Add ConversationSummaryMemory for longer chat sessions
- [ ] Deploy to Streamlit Community Cloud

## ⚠️ Notes

- `langchain==0.3.25` is pinned deliberately — LangChain 1.x removed the classic
  `ConversationalRetrievalChain`/`ConversationBufferMemory` APIs this project uses.
- `ingest.py` needs internet access to fetch Wikipedia articles; this is expected
  and only needs to run once (or whenever `TOPICS` changes).

## 👤 Author

**Muhammad Saqib Latif** — [@saqibjutt4544-maker](https://github.com/saqibjutt4544-maker)
Open to internship/entry-level ML and AI opportunities.

## 📄 License

This project is licensed under the MIT License — free to use, modify, and build on it.