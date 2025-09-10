# Ce bloc doit être au TOUT DÉBUT du fichier.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("✅ Passage à pysqlite3-binary réussi.")
except ImportError:
    print("⚠️ pysqlite3-binary non trouvé. Utilisation de la version par défaut de sqlite3.")

import logging
import os
import shutil
import time
import uuid

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Importez l'instance unique de votre système RAG
from graph import rag_system

# --- Configuration de la Page et du Logging ---
st.set_page_config(page_title="NewsAI - Système RAG Adaptatif", page_icon="🚀", layout="wide")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CSS Moderne avec Animations ---
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

# --- Constantes ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEMP_DIR = "temp_documents"

@st.cache_resource
def load_rag_system():
    """Charge l'instance unique du système RAG et la met en cache."""
    logging.info("--- Initialisation ou chargement du système RAG depuis le cache ---")
    return rag_system

def process_and_store_documents(uploaded_files):
    """Traite les fichiers uploadés et crée un retriever."""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    
    documents = []
    for uploaded_file in uploaded_files:
        temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        try:
            loader = PyPDFLoader(temp_path) if uploaded_file.type == "application/pdf" else TextLoader(temp_path, encoding='utf-8')
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
            logging.info(f"Loaded {len(loaded_docs)} documents from {uploaded_file.name}")
        except Exception as e:
            st.error(f"Erreur de chargement du fichier {uploaded_file.name}: {e}")
            continue
    
    if not documents:
        st.warning("Aucun document n'a pu être chargé. Veuillez vérifier les fichiers.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(splits)} chunks")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Utilise un nom de collection unique pour éviter les conflits
    collection_name = f"docs_{str(uuid.uuid4().hex)[:8]}"
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        collection_name=collection_name
    )
    
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# --- Initialisation du Système ---
try:
    rag_system_instance = load_rag_system()
    system_status = "✅ Système Opérationnel"
    system_class = "status-success"
except Exception as e:
    system_status = f"❌ Erreur: {e}"
    system_class = "status-warning"
    st.error(f"Échec de l'initialisation du système RAG: {e}")

# --- En-tête Principal ---
st.markdown("""
<div class="main-header">
    <h1>🚀 NewsAI</h1>
    <p>Système RAG Adaptatif de Nouvelle Génération</p>
</div>
""", unsafe_allow_html=True)

# --- Barre Latérale Moderne ---
with st.sidebar:
    st.markdown('<h2 style="color: white; text-align: center;">🎛️ Centre de Contrôle</h2>', unsafe_allow_html=True)
    st.markdown(f'<div class="status-indicator {system_class}">{system_status}</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 📁 Vos Documents")
    
    uploaded_files = st.file_uploader(
        "Chargez vos documents pour une analyse personnalisée", 
        accept_multiple_files=True,
        type=['pdf', 'txt'],
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.markdown(f"**📊 {len(uploaded_files)} fichier(s) sélectionné(s)**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Traiter", use_container_width=True):
                with st.spinner("🔄 Traitement des documents..."):
                    try:
                        retriever = process_and_store_documents(uploaded_files)
                        if retriever:
                            st.session_state.retriever = retriever
                            st.session_state.document_names = [f.name for f in uploaded_files]
                            st.success("✅ Documents traités!")
                            st.balloons()
                    except Exception as e:
                        st.error(f"❌ Erreur: {e}")
        
        with col2:
            if st.button("🗑️ Effacer", use_container_width=True):
                keys_to_delete = ['retriever', 'document_names', 'messages']
                for key in keys_to_delete:
                    if key in st.session_state:
                        del st.session_state[key]
                if os.path.exists(TEMP_DIR):
                    shutil.rmtree(TEMP_DIR)
                st.success("🗑️ Documents et chat effacés!")
                st.rerun()

    if 'document_names' in st.session_state:
        st.markdown("### 📚 Documents Actifs")
        for doc_name in st.session_state.document_names:
            st.markdown(f" - `{doc_name}`")

    st.markdown("---")
    
    st.markdown("### ⚡ Actions Rapides")
    if st.button("🧹 Vider le Cache", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache vidé! Redémarrage...")
        time.sleep(1)
        st.rerun()

# --- Interface de Chat Principale ---
# Initialisation du thread_id pour la mémoire conversationnelle
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Message de bienvenue
if not st.session_state.messages:
    st.markdown('''
    <div style="background: rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 2rem; margin: 1rem 0; backdrop-filter: blur(10px);">
        <h3>👋 Bienvenue sur NewsAI</h3>
        <p>Chargez des documents pour une analyse personnalisée, ou posez des questions générales en utilisant la recherche web.</p>
    </div>
    ''', unsafe_allow_html=True)

# Affichage des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="chat-message">{message["content"]}</div>', unsafe_allow_html=True)
        if "details" in message:
            with st.expander("🔍 Détails du Traitement"):
                details = message["details"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Temps", f"{details.get('Temps de traitement (s)', 0)}s")
                with col2:
                    st.metric("🔄 Réécritures", details.get('Nombre de réécritures', 0))
                with col3:
                    st.metric("📄 Documents", details.get('Documents pertinents utilisés', 0))

# Zone de saisie
if prompt := st.chat_input("💭 Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-message">{prompt}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        response_data = {}
        retriever_obj = st.session_state.get("retriever")

        # Configuration pour le passage du thread_id
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        if retriever_obj:
            st.markdown('<div class="status-indicator status-info">📚 Utilisation de vos documents...</div>', unsafe_allow_html=True)
            logging.info(f"✅ Retriever trouvé dans session state. Type: {type(retriever_obj)}")
            
            # Test the retriever before using it
            try:
                test_docs = retriever_obj.invoke("test")
                logging.info(f"✅ Retriever test successful - can retrieve {len(test_docs)} documents")
            except Exception as e:
                logging.error(f"❌ Retriever test failed: {e}")
                st.error(f"Erreur avec le retriever: {e}")
            
            with st.spinner("Traitement avec NewsAI..."):
                response_data = rag_system_instance.ask_question(prompt, retriever=retriever_obj, config=config)
        else:
            st.markdown('<div class="status-indicator status-info">🌐 Utilisation des connaissances générales...</div>', unsafe_allow_html=True)
            logging.warning("⚠️ Pas de retriever dans session state. Mode général.")
            with st.spinner("Analyse avec NewsAI..."):
                response_data = rag_system_instance.ask_question(prompt, retriever=None, config=config)

        if response_data.get("success", False):
            answer = response_data["answer"]
            st.markdown(f'<div class="chat-message">{answer}</div>', unsafe_allow_html=True)
            details = {
                "Temps de traitement (s)": response_data.get("processing_time", 0),
                "Nombre de réécritures": response_data.get("query_rewrites", 0),
                "Documents pertinents utilisés": len(response_data.get("documents", [])),
            }
            st.session_state.messages.append({"role": "assistant", "content": answer, "details": details})
        else:
            error_msg = response_data.get("answer", "Une erreur inconnue est survenue.")
            st.error(f"❌ {error_msg}")
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        st.rerun()
