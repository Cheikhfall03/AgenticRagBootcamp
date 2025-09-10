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

# --- CSS (Optionnel, à garder tel quel) ---
st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Gardez votre CSS ici

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
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"Erreur de chargement du fichier {uploaded_file.name}: {e}")
    
    if not documents:
        st.warning("Aucun document n'a pu être chargé. Veuillez vérifier les fichiers.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Utilise un nom de collection unique pour éviter les conflits
    collection_name = f"docs_{str(uuid.uuid4().hex)[:8]}"
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, collection_name=collection_name)
    
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# --- Initialisation du Système ---
rag_system_instance = load_rag_system()

# --- Interface Principale ---
st.title("🚀 NewsAI - Système RAG Adaptatif")

# --- Barre Latérale ---
with st.sidebar:
    st.header("🎛️ Centre de Contrôle")
    uploaded_files = st.file_uploader("Chargez vos documents", accept_multiple_files=True, type=['pdf', 'txt'])

    if uploaded_files:
        if st.button("Traiter les Documents", use_container_width=True, type="primary"):
            with st.spinner("Traitement des documents..."):
                try:
                    retriever = process_and_store_documents(uploaded_files)
                    st.session_state.retriever = retriever
                    st.session_state.document_names = [f.name for f in uploaded_files]
                    st.success("Documents traités avec succès !")
                    st.balloons()
                except Exception as e:
                    st.error(f"Une erreur est survenue: {e}")
    
    if 'retriever' in st.session_state:
        st.success(f"{len(st.session_state.document_names)} document(s) actifs.")
        if st.button("Effacer les documents", use_container_width=True):
            for key in ['retriever', 'document_names', 'messages']:
                if key in st.session_state:
                    del st.session_state[key]
            if os.path.exists(TEMP_DIR):
                shutil.rmtree(TEMP_DIR)
            st.success("Documents et conversation effacés.")
            st.rerun()

# --- Interface de Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "details" in message:
            with st.expander("Détails du traitement"):
                st.json(message["details"])

if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Réflexion en cours..."):
            retriever_obj = st.session_state.get("retriever")
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            response_data = rag_system_instance.ask_question(prompt, retriever=retriever_obj, config=config)

            if response_data.get("success"):
                answer = response_data["answer"]
                message_placeholder.markdown(answer)
                details = {
                    "Temps de traitement (s)": response_data.get("processing_time"),
                    "Nombre de réécritures": response_data.get("query_rewrites"),
                    "Documents pertinents utilisés": response_data.get("documents_used"),
                }
                st.session_state.messages.append({"role": "assistant", "content": answer, "details": details})
            else:
                error_msg = response_data.get("answer", "Une erreur inconnue est survenue.")
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
