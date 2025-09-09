import os
os.environ["USER_AGENT"] = "FinalRagBootcamp/1.0"
os.environ["CHROMA_TELEMETRY"] = "FALSE"  # Désactive la télémétrie ChromaDB

import chromadb
# Patch telemetry to avoid argument errors
chromadb.telemetry.capture = lambda *args, **kwargs: None

from pathlib import Path
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    WebBaseLoader
)
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, DocumentCompressorPipeline, CrossEncoderReranker
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration des embeddings et LLM
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0.0,
)

def load_documents(file_paths: List[str] = None, urls: List[str] = None):
    """
    Charge les documents depuis des fichiers uploadés ou des URLs
    """
    docs_list = []
    
    # Chargement des fichiers uploadés
    if file_paths:
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Fichier non trouvé: {file_path}")
                continue
                
            file_extension = Path(file_path).suffix.lower()
            
            try:
                if file_extension == '.pdf':
                    loader = PyPDFLoader(file_path)
                elif file_extension == '.txt':
                    loader = TextLoader(file_path, encoding='utf-8')
                elif file_extension in ['.docx', '.doc']:
                    loader = Docx2txtLoader(file_path)
                elif file_extension == '.csv':
                    loader = CSVLoader(file_path)
                elif file_extension in ['.xlsx', '.xls']:
                    loader = UnstructuredExcelLoader(file_path)
                else:
                    print(f"Type de fichier non supporté: {file_extension}")
                    continue
                
                docs = loader.load()
                docs_list.extend(docs)
                print(f"Chargé {len(docs)} documents depuis {file_path}")
                
            except Exception as e:
                print(f"Erreur lors du chargement de {file_path}: {e}")
    
    # Chargement des URLs (optionnel)
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
    """
    Découpe les documents en utilisant le semantic chunking
    """
    semantic_chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
    
    doc_splits = semantic_chunker.split_documents(docs)
    print(f"Documents découpés en {len(doc_splits)} chunks sémantiques")
    
    return doc_splits

def split_documents_recursive(docs):
    """
    Découpe récursive - méthode alternative
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],  # Default separators
    )
    doc_splits = text_splitter.split_documents(docs)
    print(f"Documents découpés en {len(doc_splits)} chunks récursifs")
    return doc_splits

def create_hybrid_retriever(doc_splits, vectorstore):
    """
    Crée un retriever hybride combinant recherche vectorielle et BM25
    """
    vector_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    bm25_retriever = BM25Retriever.from_documents(doc_splits)
    bm25_retriever.k = 10
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )
    return ensemble_retriever

def create_reranker():
    """
    Crée un reranker basé sur cross-encoder
    """
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model)
    return compressor

def create_contextual_compressor():
    """
    Crée un compresseur contextuel qui extrait les parties pertinentes
    """
    compressor = LLMChainExtractor.from_llm(llm=llm)
    return compressor

def create_advanced_retriever(doc_splits, vectorstore, use_reranker=True, use_compression=True):
    """
    Crée un retriever avancé avec hybrid search, reranking et compression
    """
    base_retriever = create_hybrid_retriever(doc_splits, vectorstore)
    compressors = []
    
    if use_reranker:
        compressors.append(create_reranker())
        print("✅ Reranker ajouté")
    
    if use_compression:
        compressors.append(create_contextual_compressor())
        print("✅ Compression contextuelle ajoutée")
    
    if compressors:
        pipeline_compressor = DocumentCompressorPipeline(transformers=compressors)
        compressed_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor,
            base_retriever=base_retriever
        )
        print("✅ Retriever hybride avec reranking et compression créé")
        return compressed_retriever
    else:
        print("✅ Retriever hybride simple créé")
        return base_retriever

def create_vectorstore(uploaded_files: List[str] = None, 
                      use_semantic_chunking: bool = True,
                      use_hybrid_search: bool = True,
                      use_reranker: bool = True,
                      use_compression: bool = True):
    """
    Crée le vector store avec les documents uploadés et retriever avancé
    """
    # --- CORRECTION APPLIQUÉE ICI ---
    # Si aucun fichier n'est uploadé (cas du démarrage de l'app),
    # on charge des documents par défaut.
    if not uploaded_files:
        print("Aucun fichier uploadé, utilisation des URLs par défaut pour l'initialisation.")
        urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]
        documents = load_documents(urls=urls)
    else:
        # Si des fichiers sont fournis, on les charge.
        documents = load_documents(file_paths=uploaded_files)
    
    if not documents:
        raise ValueError("Aucun document n'a pu être chargé.")
    
    if use_semantic_chunking:
        doc_splits = split_documents_semantic(documents)
    else:
        doc_splits = split_documents_recursive(documents)
    
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"Vector store créé avec {len(doc_splits)} chunks")
    
    if use_hybrid_search or use_reranker or use_compression:
        retriever_instance = create_advanced_retriever(
            doc_splits=doc_splits,
            vectorstore=vectorstore,
            use_reranker=use_reranker,
            use_compression=use_compression
        )
    else:
        retriever_instance = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        print("✅ Retriever simple créé")
    
    return vectorstore, retriever_instance

# --- Initialisation du retriever par défaut au démarrage ---
try:
    print("🚀 Initialisation du système RAG par défaut...")
    
    # On initialise un retriever global qui sera utilisé quand aucun fichier n'est uploadé
    vectorstore, retriever = create_vectorstore(
        uploaded_files=None, # On appelle sans fichier pour charger les URLs par défaut
        use_semantic_chunking=True,
        use_hybrid_search=True,
        use_reranker=True,
        use_compression=True
    )
    
    print("🚀 Système RAG par défaut initialisé avec succès!")
    
except Exception as e:
    print(f"❌ Erreur lors de l'initialisation : {e}")
    retriever = None
    print("⚠️ Retriever par défaut non disponible.")