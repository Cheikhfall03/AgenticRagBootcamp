# ingestion.py (Version Corrigée)
import os
os.environ["USER_AGENT"] = "FinalRagBootcamp/1.0"
os.environ["CHROMA_TELEMETRY"] = "FALSE"

import chromadb
# Patch telemetry to avoid argument errors
chromadb.telemetry.capture = lambda *args, **kwargs: None

from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, CrossEncoderReranker
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration partagée ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_documents(file_paths: List[str] = None, urls: List[str] = None):
    docs_list = []
    if file_paths:
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Fichier non trouvé: {file_path}")
                continue
            
            file_extension = Path(file_path).suffix.lower()
            try:
                if file_extension == '.pdf': loader = PyPDFLoader(file_path)
                elif file_extension == '.txt': loader = TextLoader(file_path, encoding='utf-8')
                elif file_extension in ['.docx', '.doc']: loader = Docx2txtLoader(file_path)
                elif file_extension == '.csv': loader = CSVLoader(file_path)
                elif file_extension in ['.xlsx', '.xls']: loader = UnstructuredExcelLoader(file_path)
                else:
                    print(f"Type de fichier non supporté: {file_extension}")
                    continue
                docs = loader.load()
                docs_list.extend(docs)
                print(f"Chargé {len(docs)} documents depuis {file_path}")
            except Exception as e:
                print(f"Erreur lors du chargement de {file_path}: {e}")
    
    if urls:
        try:
            web_docs = [WebBaseLoader(url).load() for url in urls]
            for doc_list in web_docs:
                docs_list.extend(doc_list)
            print(f"Chargé {len(web_docs)} documents web")
        except Exception as e:
            print(f"Erreur lors du chargement web: {e}")
    return docs_list

def split_documents_semantic(docs):
    semantic_chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
    doc_splits = semantic_chunker.split_documents(docs)
    print(f"Documents découpés en {len(doc_splits)} chunks sémantiques")
    return doc_splits

def create_advanced_retriever(doc_splits: List[Any], vectorstore: Chroma) -> ContextualCompressionRetriever:
    """Crée un retriever avancé avec recherche hybride et reranking."""
    vector_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    bm25_retriever = BM25Retriever.from_documents(doc_splits)
    bm25_retriever.k = 10
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )
    
    reranker_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=reranker_model, top_n=5) # <--- Re-ranke et garde le top 5
    
    pipeline_compressor = DocumentCompressorPipeline(transformers=[compressor])
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=ensemble_retriever
    )
    print("✅ Retriever avancé (hybride + reranker) créé.")
    return compression_retriever

def create_retriever_from_files(uploaded_files: List[str]) -> Any:
    """
    Crée un retriever complet à partir d'une liste de chemins de fichiers.
    C'est la fonction à appeler depuis l'interface Streamlit.
    """
    if not uploaded_files:
        raise ValueError("Aucun fichier fourni pour créer le retriever.")
        
    documents = load_documents(file_paths=uploaded_files)
    if not documents:
        raise ValueError("Aucun document n'a pu être chargé à partir des fichiers fournis.")
        
    doc_splits = split_documents_semantic(documents)
    
    # Utiliser un vectorstore en mémoire pour les documents uploadés par session
    vectorstore = Chroma.from_documents(documents=doc_splits, embedding=embeddings)
    
    print(f"Vector store de session créé avec {len(doc_splits)} chunks")
    
    return create_advanced_retriever(doc_splits, vectorstore)

def initialize_default_retriever() -> Any:
    """
    Crée et renvoie le retriever par défaut basé sur des URLs prédéfinies.
    Cette fonction est appelée une seule fois au démarrage du système.
    """
    print("🚀 Initialisation du retriever par défaut...")
    default_urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    documents = load_documents(urls=default_urls)
    if not documents:
        raise ConnectionError("Impossible de charger les documents par défaut. Vérifiez la connexion internet.")
        
    doc_splits = split_documents_semantic(documents)
    
    # Persister le vectorstore par défaut pour ne pas le reconstruire à chaque fois
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding=embeddings,
        persist_directory="./default_chroma_db"
    )
    print(f"Vector store par défaut créé et persisté avec {len(doc_splits)} chunks")
    
    retriever = create_advanced_retriever(doc_splits, vectorstore)
    print("✅ Retriever par défaut initialisé avec succès !")
    return retriever
