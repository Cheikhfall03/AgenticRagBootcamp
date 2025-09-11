import time
import traceback
from typing import Dict, Any, Optional, Iterator
from dotenv import load_dotenv
import os
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# --- Import des composants du graphe ---
from chains.answer_grader import answer_grader
from chains.retriever_grader import retrieval_grader
from chains.router_query import question_router, RouteQuery
from chains.hallucination_grader import hallucination_grader
from nodes.generate import generate
from nodes.query_rewrite import query_rewrite
from nodes.web_search import web_search
from Node_constant import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH, QUERY_REWRITE, ROUTE_QUESTION
from state import GraphState

# --- Import de l'initialiseur du retriever par défaut ---
from ingestion.ingestion import initialize_default_retriever

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("❌ TAVILY_API_KEY environment variable is not set! Add it to your .env file.")


class AdaptiveRAGSystem:
    def __init__(self):
        self.workflow = StateGraph(GraphState)
        try:
            self.default_retriever = initialize_default_retriever()
        except Exception as e:
            print(f"❌ ERREUR CRITIQUE: Impossible d'initialiser le retriever par défaut: {e}")
            self.default_retriever = None

        # ### SUPPRIMÉ ### : self.current_retriever n'est plus nécessaire ici.
        self._setup_workflow()
        memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=memory)
        print("✅ Graphe LangGraph compilé avec succès.")

    def _route_question(self, state: GraphState) -> Dict[str, Any]:
        # ... (Pas de changement dans cette méthode)
        print("---NŒUD: ROUTAGE DE LA QUESTION---")
        question = state["question"]
        try:
            source: RouteQuery = question_router.invoke({"question": question})
            print(f"📌 Décision de routage brute: {source}")
            datasource = str(source.datasource).strip().lower()
            
            if datasource == WEBSEARCH:
                print("➡️ Décision: La question nécessite une recherche web.")
                return {"next": WEBSEARCH}
            elif datasource == RETRIEVE:
                print("➡️ Décision: La question concerne les documents fournis.")
                return {"next": RETRIEVE}
            else:
                print(f"⚠️ Datasource inconnue ({source.datasource}). Fallback sur vectorstore.")
                return {"next": RETRIEVE}
        except Exception as e:
            print(f"⚠️ Erreur de routage pour la question '{question}': {e}")
            print("➡️ Fallback: récupération de documents.")
            return {"next": RETRIEVE}

# In graph.py

 # In graph.py

    def _retrieve_documents(self, state: GraphState) -> Dict[str, Any]:
        print("---NŒUD: RÉCUPÉRATION DE DOCUMENTS---")
        question = state["question"]
        retriever = state.get("retriever")
    
        if retriever is None:
            print("⚠️ Aucun retriever n'est disponible. Retourne une liste vide.")
            # Ensure retriever is None in the output state
            return {"documents": [], "question": question, "retriever": None}
    
        try:
            print(f"🔎 Utilisation du retriever: {type(retriever)}")
            documents = retriever.invoke(question)
            print(f"✅ {len(documents)} document(s) récupéré(s).")
            
            # THIS IS THE FIX 👇
            # After using the retriever, set it to None in the state. This
            # prevents the checkpointer from trying to serialize it.
            return {"documents": documents, "question": question, "retriever": None}
    
        except Exception as e:
            print(f"❌ Erreur lors de la récupération de documents: {e}")
            # Also ensure retriever is None on error
            return {"documents": [], "question": question, "retriever": None}
    def _grade_documents(self, state: GraphState) -> Dict[str, Any]:
        print("---NŒUD: ÉVALUATION DE LA PERTINENCE DES DOCUMENTS---")
        question = state["question"]
        documents = state["documents"]
        if not documents:
            return {"documents": [], "question": question}
        filtered_docs = []
        for d in documents:
            try:
                doc_text = getattr(d, "page_content", str(d))
                score = retrieval_grader.invoke({"question": question, "document": doc_text})
                if getattr(score, "binary_score", False):
                    print("✅ Document pertinent.")
                    filtered_docs.append(d)
                else:
                    print("⛔ Document non pertinent.")
            except Exception as e:
                print(f"⚠️ Erreur d'évaluation, inclusion par défaut: {e}")
                filtered_docs.append(d)
        return {"documents": filtered_docs, "question": question}

    def _decide_to_generate(self, state: GraphState) -> str:
        print("---NŒUD: DÉCISION POST-RÉCUPÉRATION---")
        if state["documents"]:
            print("✅ Documents pertinents trouvés. Passage à la génération.")
            return GENERATE
        else:
            if state["query_rewrite_count"] < 1:
                print("⛔ Aucun document pertinent. Tentative de réécriture de la question.")
                return QUERY_REWRITE
            else:
                print("⛔ Échec après réécriture. Passage à la recherche web comme dernier recours.")
                return WEBSEARCH

    def _grade_generation(self, state: GraphState) -> str:
        print("---NŒUD: ÉVALUATION DE LA GÉNÉRATION---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        if not generation:
            print("⛔ Génération vide. Fin.")
            return END

        is_web_search_result = all('source' in doc.metadata for doc in documents if hasattr(doc, 'metadata'))

        if is_web_search_result:
            print("⚠️ Réponse issue du web → Vérification de l'utilité.")
            answer_score = answer_grader.invoke({"question": question, "generation": generation})
            return END if getattr(answer_score, "binary_score", False) else END

        print("---VÉRIFICATION DES HALLUCINATIONS---")
        docs_texts = [getattr(d, "page_content", str(d)) for d in documents]
        full_context = "\n\n---\n\n".join(docs_texts)
        try:
            hallucination_score = hallucination_grader.invoke(
                {"documents": full_context[:10000], "generation": generation}
            )
            if not getattr(hallucination_score, "binary_score", False):
                print("⛔ HALLUCINATION détectée. Tentative de re-génération.")
                return GENERATE if state["generation_count"] < 1 else END
        except Exception as e:
            print(f"⚠️ Erreur du hallucination_grader: {e}")
            return END

        print("✅ Génération ancrée. Vérification de la pertinence.")
        answer_score = answer_grader.invoke({"question": question, "generation": generation})
        return END if getattr(answer_score, "binary_score", False) else END

    def _setup_workflow(self):
        # ... (Pas de changement dans cette méthode)
        self.workflow.add_node(RETRIEVE, self._retrieve_documents)
        self.workflow.add_node(GRADE_DOCUMENTS, self._grade_documents)
        self.workflow.add_node(QUERY_REWRITE, query_rewrite)
        self.workflow.add_node(WEBSEARCH, web_search)
        self.workflow.add_node(GENERATE, generate)
        self.workflow.add_node(ROUTE_QUESTION, self._route_question)
        self.workflow.set_entry_point(ROUTE_QUESTION)
        self.workflow.add_conditional_edges(
            ROUTE_QUESTION,
            lambda state: state["next"],
            {WEBSEARCH: WEBSEARCH, RETRIEVE: RETRIEVE}
        )
        self.workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
        self.workflow.add_edge(QUERY_REWRITE, RETRIEVE)
        self.workflow.add_edge(WEBSEARCH, GENERATE)
        self.workflow.add_conditional_edges(
            GRADE_DOCUMENTS,
            self._decide_to_generate,
            {GENERATE: GENERATE, QUERY_REWRITE: QUERY_REWRITE, WEBSEARCH: WEBSEARCH}
        )
        self.workflow.add_conditional_edges(
            GENERATE,
            self._grade_generation,
            {GENERATE: GENERATE, END: END}
        )

    def run(self, question: str, retriever: Optional[Any] = None, config: Optional[Dict] = None) -> Iterator[Dict[str, Any]]:
        if not self.app:
            print("Erreur: Le système RAG n'est pas compilé.")
            return iter([])

        # ### MODIFIÉ ### : Déterminer quel retriever utiliser pour cet appel spécifique
        active_retriever = retriever if retriever is not None else self.default_retriever
        
        # ### MODIFIÉ ### : Injecter le retriever actif dans l'état initial du graphe
        initial_state = {
            "question": question,
            "query_rewrite_count": 0,
            "generation_count": 0,
            "documents": [],
            "generation": "",
            "retriever": active_retriever  # <-- LA LIGNE CLÉ
        }
        
        print(f"--- Lancement du graphe pour la question: '{question}' ---")
        return self.app.stream(initial_state, config=config)

# --- Instance unique (Singleton) pour l'application ---
rag_system = AdaptiveRAGSystem()
