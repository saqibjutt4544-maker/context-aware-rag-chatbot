"""
app.py
Streamlit UI for the context-aware RAG chatbot.

Before running this, you must have already run `python ingest.py`
at least once to create the ./vectorstore/ folder.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

st.set_page_config(page_title="Context-Aware RAG Chatbot", page_icon="💬")
st.title("💬 Context-Aware RAG Chatbot")
st.caption("Ask questions about the knowledge base. I remember our conversation as we go.")


# ---------------------------------------------------------------------------
# Setup (cached so it only runs once per session, not on every rerun)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_vectorstore():
    if not os.path.isdir(VECTORSTORE_DIR):
        st.error(
            f"No vector store found at ./{VECTORSTORE_DIR}/. "
            "Run `python ingest.py` first to build it."
        )
        st.stop()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # allow_dangerous_deserialization=True is safe here because WE created
    # this FAISS index ourselves in ingest.py -- it's not an untrusted file.
    return FAISS.load_local(
        VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
    )


def get_api_key():
    # Prefer Streamlit secrets (for deployment), fall back to env var (local dev).
    # st.secrets raises FileNotFoundError just from being accessed if no
    # secrets.toml exists at all (e.g. running locally without one) -- so we
    # have to try/except instead of relying on .get()'s default.
    key = None
    try:
        key = st.secrets.get("GROQ_API_KEY", None)
    except FileNotFoundError:
        pass
    return key or os.environ.get("GROQ_API_KEY")


def build_chain(_vectorstore, api_key):
    llm = ChatGroq(model=LLM_MODEL, api_key=api_key, temperature=0.2)
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    return chain


# ---------------------------------------------------------------------------
# Sidebar: API key input
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Setup")
    default_key = get_api_key() or ""
    api_key = st.text_input(
        "Groq API key",
        value=default_key,
        type="password",
        help="Get a free key at https://console.groq.com (no credit card needed).",
    )
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. Your question is embedded and matched against the vector store\n"
        "2. The most relevant chunks are retrieved\n"
        "3. Retrieved context + chat history + your question go to the LLM\n"
        "4. The LLM answers using that context"
    )
    if st.button("Clear conversation"):
        st.session_state.clear()
        st.rerun()

if not api_key:
    st.info("Enter your Groq API key in the sidebar to start chatting.")
    st.stop()

vectorstore = load_vectorstore()

# Rebuild chain if the API key changes, otherwise reuse across reruns
if "chain" not in st.session_state or st.session_state.get("_api_key") != api_key:
    st.session_state.chain = build_chain(vectorstore, api_key)
    st.session_state._api_key = api_key

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask something about the knowledge base...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.chain.invoke({"question": user_input})
                answer = result["answer"]
                sources = result.get("source_documents", [])
            except Exception as e:
                answer = f"Something went wrong: {e}"
                sources = []

            st.markdown(answer)

            if sources:
                with st.expander("Sources used for this answer"):
                    for i, doc in enumerate(sources, 1):
                        title = doc.metadata.get("title", doc.metadata.get("source", f"Source {i}"))
                        st.markdown(f"**{i}. {title}**")
                        st.caption(doc.page_content[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})