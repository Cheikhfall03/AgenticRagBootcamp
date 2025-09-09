# streamlit_app.py

import streamlit as st
import time
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import shutil
import logging
import uuid

# --- Configuration and Custom CSS ---
st.set_page_config(
    page_title="NewsAI - Adaptive RAG System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging to be visible in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Modern, professional CSS with animations and glassmorphism
st.markdown("""
<style>
    /* CSS Variables for Color Consistency */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --accent-color: #f093fb;
        --success-color: #4ecdc4;
        --warning-color: #feca57;
        --error-color: #ff6b6b;
        --text-dark: #2c3e50;
        --text-light: #ecf0f1;
        --bg-dark: #1e2126;
        --bg-light: #f8fafc;
        --card-bg: rgba(255, 255, 255, 0.05);
        --border-radius: 16px;
        --shadow-light: 0 4px 20px rgba(102, 126, 234, 0.1);
        --shadow-medium: 0 8px 32px rgba(102, 126, 234, 0.2);
        --gradient-primary: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        --gradient-accent: linear-gradient(135deg, var(--accent-color) 0%, var(--primary-color) 100%);
    }
    .main { padding: 2rem 1rem; }
    .main-header {
        background: var(--gradient-primary); padding: 3rem 2rem; border-radius: var(--border-radius);
        text-align: center; margin-bottom: 2rem; box-shadow: var(--shadow-medium);
        position: relative; overflow: hidden;
    }
    .main-header h1 { color: white; font-size: 3rem; font-weight: 700; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-top: 1rem; }
    .glass-card {
        background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2); border-radius: var(--border-radius);
        padding: 2rem; margin: 1rem 0; box-shadow: var(--shadow-light);
    }
    .stButton > button {
        background: var(--gradient-accent); color: white; border: none;
        border-radius: var(--border-radius); padding: 0.75rem 1.5rem; font-weight: 600;
        transition: all 0.3s ease; box-shadow: var(--shadow-light);
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: var(--shadow-medium); }
    .chat-message {
        background: var(--card-bg); border-radius: var(--border-radius); padding: 1.5rem;
        margin: 1rem 0; border-left: 4px solid var(--primary-color);
        backdrop-filter: blur(10px); animation: slideIn 0.5s ease-out;
    }
    @keyframes slideIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    .status-indicator {
        display: inline-flex; align-items: center; padding: 0.5rem 1rem;
        border-radius: 50px; font-size: 0.9rem; font-weight: 600; margin: 0.5rem 0; color: white;
    }
    .status-success { background: linear-gradient(135deg, var(--success-color), #48cae4); }
    .status-warning { background: linear-gradient(135deg, var(--warning-color), #ffb347); }
    .status-info { background: linear-gradient(135deg, var(--primary-color), var(--accent-color)); }
    .metric-card {
        background: var(--gradient-accent); border-radius: var(--border-radius);
        padding: 1.5rem; text-align: center; color: white; margin: 0.5rem;
        box-shadow: var(--shadow-light);
    }
    .metric-value { font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }
    .metric-label { font-size: 0.9rem; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEMP_DIR = "temp_documents"

@st.cache_resource
def load_rag_system():
    """Load the main RAG system once and cache it."""
    logging.info("--- Initializing or loading cached RAG System ---")
    # This ensures the latest code from graph.py is used when the cache is cleared.
    from graph import rag_system
    return rag_system

def process_and_store_documents(uploaded_files):
    """Processes uploaded files and creates a retriever."""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    
    documents = []
    
    for uploaded_file in uploaded_files:
        temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        try:
            loader = PyPDFLoader(temp_path) if uploaded_file.type == "application/pdf" else TextLoader(temp_path, encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)
            logging.info(f"Loaded {len(docs)} documents from {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
            continue
    
    if not documents:
        raise ValueError("No documents were successfully loaded")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(splits)} chunks")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    collection_name = f"docs_{str(uuid.uuid4())[:8]}"
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        collection_name=collection_name
    )
    
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# --- System Initialization ---
try:
    rag_system = load_rag_system()
    system_status = "✅ System Operational"
    system_class = "status-success"
except Exception as e:
    system_status = f"❌ Error: {e}"
    system_class = "status-warning"
    st.error(f"Failed to initialize RAG system: {e}")

# --- Main Header ---
st.markdown("""
<div class="main-header">
    <h1>🚀 NewsAI</h1>
    <p>Next-Generation Adaptive RAG System</p>
</div>
""", unsafe_allow_html=True)

# --- Modern Sidebar ---
with st.sidebar:
    st.markdown('<h2 style="color: white; text-align: center;">🎛️ Control Center</h2>', unsafe_allow_html=True)
    st.markdown(f'<div class="status-indicator {system_class}">{system_status}</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 📁 Your Documents")
    
    uploaded_files = st.file_uploader(
        "Upload your documents for personalized analysis", 
        accept_multiple_files=True,
        type=['pdf', 'txt'],
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.markdown(f"**📊 {len(uploaded_files)} file(s) selected**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Process", use_container_width=True):
                with st.spinner("🔄 Processing documents..."):
                    try:
                        retriever = process_and_store_documents(uploaded_files)
                        st.session_state.retriever = retriever
                        st.session_state.document_names = [f.name for f in uploaded_files]
                        st.success("✅ Documents processed!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                keys_to_delete = ['retriever', 'document_names', 'messages']
                for key in keys_to_delete:
                    if key in st.session_state:
                        del st.session_state[key]
                if os.path.exists(TEMP_DIR):
                    shutil.rmtree(TEMP_DIR)
                st.success("🗑️ Documents & chat cleared!")
                st.rerun()

    if 'document_names' in st.session_state:
        st.markdown("### 📚 Active Documents")
        for doc_name in st.session_state.document_names:
            st.markdown(f" - `{doc_name}`")

    st.markdown("---")
    
    st.markdown("### ⚡ Quick Actions")
    if st.button("🧹 Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared! Restarting...")
        time.sleep(1)
        st.rerun()

# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown('<div class="glass-card"><h3>👋 Welcome to NewsAI</h3><p>Upload documents for analysis, or ask general questions using web search.</p></div>', unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="chat-message">{message["content"]}</div>', unsafe_allow_html=True)
        if "details" in message:
            with st.expander("🔍 Processing Details"):
                details = message["details"]
                st.write(details)

# --- Input Area ---
if prompt := st.chat_input("💭 Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-message">{prompt}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        response_data = {}
        retriever_obj = st.session_state.get("retriever")

        if retriever_obj:
            st.markdown('<div class="status-indicator status-info">📚 Using your documents...</div>', unsafe_allow_html=True)
            logging.info(f"✅ Retriever found in session state. Type: {type(retriever_obj)}. Invoking RAG.")
            with st.spinner("Processing with NewsAI..."):
                response_data = rag_system.ask_question(prompt, retriever=retriever_obj)
        else:
            st.markdown('<div class="status-indicator status-info">🌐 Using general knowledge...</div>', unsafe_allow_html=True)
            logging.warning("⚠️ No retriever in session state. Falling back to general mode.")
            with st.spinner("Analyzing with NewsAI..."):
                response_data = rag_system.ask_question(prompt, retriever=None)

        if response_data.get("success", False):
            answer = response_data["answer"]
            st.markdown(f'<div class="chat-message">{answer}</div>', unsafe_allow_html=True)
            details = {
                "Time (s)": response_data.get("processing_time", 0),
                "Rewrites": response_data.get("query_rewrites", 0),
                "Documents Used": len(response_data.get("documents", [])),
            }
            st.session_state.messages.append({"role": "assistant", "content": answer, "details": details})
        else:
            error_msg = response_data.get("answer", "An unknown error occurred.")
            st.error(f"❌ {error_msg}")
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        st.rerun()

