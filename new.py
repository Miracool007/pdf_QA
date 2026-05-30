import streamlit as st
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

# ── Environment ────────────────────────────────────────────────────────────────
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

groq_api_key = os.getenv("GROQ_API_KEY")

# ── Model initialisation ───────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

embeddings = load_embeddings()
llm = load_llm()

# ── Session-state bootstrap ────────────────────────────────────────────────────
if "store" not in st.session_state:
    st.session_state.store = {}          

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None  

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# ── Helper: memory per session ─────────────────────────────────────────────────
def get_memory(session_id: str) -> list:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = []
    return st.session_state.store[session_id]

# ── Helper: build / update vectorstore ────────────────────────────────────────
def build_vectorstore(files) -> Chroma:
    documents = []

    for uploaded_file in files:
        if uploaded_file.name in st.session_state.processed_files:
            continue  

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())
            st.session_state.processed_files.add(uploaded_file.name)
        finally:
            os.unlink(tmp_path)  # clean up temp file

    if not documents:
        return st.session_state.vectorstore  

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = Chroma.from_documents(
            documents=splits, embedding=embeddings
        )
    else:
        st.session_state.vectorstore.add_documents(splits)

    return st.session_state.vectorstore

# ── Helper: RAG response ───────────────────────────────────────────────────────
def get_response(query: str, memory: list, vectorstore: Chroma) -> str:
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    docs = retriever.invoke(query)
    context = "\n\n------\n\n".join(doc.page_content for doc in docs)

    system_prompt = (
        "You are a helpful assistant for question-answering tasks. "
        "Use the retrieved document context below to answer the user's question. "
        "If the context does not contain enough information, say so politely. "
        "Aim for clear, thorough answers that help the user fully understand the topic."
    )

    conversation_history = "\n".join(
        f"User: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}"
        for m in memory
    )

    full_prompt = (
        f"{system_prompt}\n\n"
        f"Context from document:\n{context}\n\n"
        f"Conversation so far:\n{conversation_history}\n\n"
        f"User: {query}\nAI:"
    )

    memory.append(HumanMessage(content=query))

    response = llm.invoke([HumanMessage(content=full_prompt)])

    memory.append(AIMessage(content=response.content))

    return response.content

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("📄 Conversational RAG — PDF Q&A")
st.markdown(
    """
    Welcome! This app lets you **chat with your PDF documents** using AI.  
    Upload one or more PDFs and ask any question — the assistant retrieves the 
    most relevant content from your documents and generates clear, context-aware answers.  
    Your conversation history is preserved throughout the session, so follow-up questions work naturally.
    """
)

session_id = st.text_input("Session ID", value="default_session")

uploaded_files = st.file_uploader(
    "Choose PDF file(s)", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Embedding documents…"):
        vectorstore = build_vectorstore(uploaded_files)

    if vectorstore:
        st.success(
            f"✅ {len(st.session_state.processed_files)} file(s) ready: "
            + ", ".join(st.session_state.processed_files)
        )

        memory = get_memory(session_id)
        for msg in memory:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.write(msg.content)

        query = st.chat_input("Ask a question about your documents…")
        if query:
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    response = get_response(query, memory=memory, vectorstore=vectorstore)
                st.write(response)
                
