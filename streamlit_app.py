# streamlit_app.py (Version Complète et Finalisée)

# This block must be at the VERY BEGINNING of the file.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("✅ Switched to pysqlite3-binary successfully.")
except ImportError:
    print("⚠️ pysqlite3-binary not found. Using default sqlite3.")

import logging
import os
import shutil
import time
import uuid

import streamlit as st

# Import the unique instance of your RAG system from graph.py
from graph import AdaptiveRAGSystem 
# Import the retriever creation function from ingestion.py
from ingestion.ingestion import create_retriever_from_files

# --- Page and Logging Configuration ---
st.set_page_config(page_title="NewsAI - Adaptive RAG System", page_icon="🚀", layout="wide")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Modern CSS with Animations ---
st.markdown("""
<style>
    .main { padding: 2rem 1rem; }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem; border-radius: 16px; text-align: center;
        margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    }
    .main-header h1 { color: white; font-size: 3rem; font-weight: 700; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-top: 1rem; }
    .chat-message {
        background: rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 1.5rem;
        margin: 1rem 0; border-left: 4px solid #667eea;
        backdrop-filter: blur(10px); animation: slideIn 0.5s ease-out;
    }
    @keyframes slideIn { 
        from { opacity: 0; transform: translateY(20px); } 
        to { opacity: 1; transform: translateY(0); } 
    }
    .status-indicator {
        display: inline-flex; align-items: center; padding: 0.5rem 1rem;
        border-radius: 50px; font-size: 0.9rem; font-weight: 600; 
        margin: 0.5rem 0; color: white;
    }
    .status-success { background: linear-gradient(135deg, #4ecdc4, #48cae4); }
    .status-warning { background: linear-gradient(135deg, #feca57, #ffb347); }
    .status-info { background: linear-gradient(135deg, #667eea, #f093fb); }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
TEMP_DIR = "temp_documents"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# --- System Initialization ---
try:
    rag_system = AdaptiveRAGSystem()
    rag_system_instance = rag_system
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
        "Upload your documents for custom analysis", 
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
                        file_paths = []
                        for uploaded_file in uploaded_files:
                            temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getvalue())
                            file_paths.append(temp_path)
                        
                        retriever = create_retriever_from_files(file_paths)
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
                    os.makedirs(TEMP_DIR)
                st.success("🗑️ Documents and chat cleared!")
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
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome message
if not st.session_state.messages:
    st.markdown('''
    <div style="background: rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 2rem; margin: 1rem 0; backdrop-filter: blur(10px);">
        <h3>👋 Welcome to NewsAI</h3>
        <p>Upload documents for personalized analysis, or ask general questions using the default knowledge base.</p>
    </div>
    ''', unsafe_allow_html=True)

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="chat-message">{message["content"]}</div>', unsafe_allow_html=True)

# Chat input and streaming response
if prompt := st.chat_input("💭 Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-message">{prompt}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        # Get the custom retriever from the session, if it exists
        retriever_for_this_query = st.session_state.get("retriever")

        if retriever_for_this_query:
            st.markdown('<div class="status-indicator status-info">📚 Using your documents...</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-indicator status-info">🌐 Using general knowledge...</div>', unsafe_allow_html=True)

        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        # Use the clean .run() method from the RAG system class
        # This method handles all the internal logic of selecting the retriever
        response_stream = rag_system_instance.run(
            prompt,
            retriever=retriever_for_this_query,
            config=config
        )
        
        # Generator function to correctly process the LangGraph stream
        def stream_rag_response():
            for event in response_stream:
                # The stream yields a dictionary: {node_name: node_output}
                # Check if the 'generate' node is the key for the current event
                if "generate" in event:
                    # If so, get the "generation" value from the node's output
                    generation_chunk = event["generate"].get("generation")
                    if generation_chunk:
                        yield generation_chunk

        # Use st.write_stream to display the response from the generator
        response_container = st.empty()
        full_response = response_container.write_stream(stream_rag_response)
        
        # Display the final message and save it to history
        if full_response:
             response_container.markdown(f'<div class="chat-message">{full_response}</div>', unsafe_allow_html=True)
             st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
             response_container.markdown(f'<div class="chat-message">Sorry, I was unable to generate a response. Please try rephrasing your question.</div>', unsafe_allow_html=True)
             st.session_state.messages.append({"role": "assistant", "content": "Sorry, I was unable to generate a response. Please try rephrasing your question."})
