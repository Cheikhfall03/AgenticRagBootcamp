"""
Système RAG modulaire avec logique de graphe autonome.
"""
import time
import traceback
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

# Importez vos chaînes et nœuds personnalisés
from chains.answer_grader import answer_grader
from chains.hallucination_grader import hallucination_grader
from chains.retriever_grader import retrieval_grader
from chains.router_query import question_router
from Node_constant import (GRADE_DOCUMENTS, GENERATE, QUERY_REWRITE, RETRIEVE,
                           WEBSEARCH)
from nodes.generate import generate as generate_node
from nodes.query_rewrite import query_rewrite as query_rewrite_node
from nodes.web_search import web_search as web_search_node

load_dotenv()

class GraphState(TypedDict):
    """Représente l'état de notre graphe. Ne contient que des données sérialisables."""
    question: str
    generation: str
    documents: List[Any]
    query_rewrite_count: int
    generation_count: int

class AdaptiveRAGSystem:
    """Système RAG adaptatif et modulaire."""

    def __init__(self):
        self.app = None
        self.retriever = None
        self._setup_workflow()

    def _retrieve_documents(self, state: GraphState) -> Dict[str, Any]:
        print("---NŒUD: RÉCUPÉRER LES DOCUMENTS---")
        question = state["question"]
        retriever = self.retriever

        documents = []
        if retriever is not None:
            print("▶️ Utilisation du retriever des documents uploadés")
            try:
                documents = retriever.invoke(question)
                print(f"✅ {len(documents)} documents récupérés")
            except Exception as e:
                print(f"❌ Erreur lors de la récupération : {e}")
                traceback.print_exc()
        else:
            print("⚠️ Aucun retriever personnalisé. Poursuite avec une liste vide.")
        
        return {"documents": documents, "question": question}

    def _grade_documents(self, state: GraphState) -> Dict[str, Any]:
        print("---NŒUD: ÉVALUER LA PERTINENCE DES DOCUMENTS---")
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []

        for d in documents:
            try:
                score = retrieval_grader.invoke(
                    {"question": question, "document": d.page_content}
                )
                if score.binary_score:
                    print("✅ Document pertinent")
                    filtered_docs.append(d)
                else:
                    print("❌ Document non pertinent")
            except Exception:
                print("⚠️ Erreur lors de l'évaluation d'un document, ignoré.")
                continue
        return {"documents": filtered_docs}

    def _query_rewrite_with_increment(self, state: GraphState) -> Dict[str, Any]:
        """Wrapper pour incrémenter le compteur avant de réécrire la requête."""
        count = state.get('query_rewrite_count', 0)
        state['query_rewrite_count'] = count + 1
        return query_rewrite_node(state)

    def _generate_with_increment(self, state: GraphState) -> Dict[str, Any]:
        """Wrapper pour incrémenter le compteur avant la génération."""
        count = state.get('generation_count', 0)
        state['generation_count'] = count + 1
        return generate_node(state)

    def _setup_workflow(self):
        self.workflow = StateGraph(GraphState)

        self.workflow.add_node(RETRIEVE, self._retrieve_documents)
        self.workflow.add_node(GRADE_DOCUMENTS, self._grade_documents)
        self.workflow.add_node(WEBSEARCH, web_search_node)
        self.workflow.add_node(GENERATE, self._generate_with_increment)
        self.workflow.add_node(QUERY_REWRITE, self._query_rewrite_with_increment)

        self.workflow.set_conditional_entry_point(self._route_question, {WEBSEARCH: WEBSEARCH, RETRIEVE: RETRIEVE})
        self.workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
        self.workflow.add_edge(QUERY_REWRITE, RETRIEVE)
        self.workflow.add_edge(WEBSEARCH, GENERATE)
        self.workflow.add_conditional_edges(GRADE_DOCUMENTS, self._decide_to_generate_or_rewrite, {QUERY_REWRITE: QUERY_REWRITE, GENERATE: GENERATE, WEBSEARCH: WEBSEARCH})
        self.workflow.add_conditional_edges(GENERATE, self._grade_generation, {"not supported": GENERATE, "useful": END, "not useful": QUERY_REWRITE, "fail": END})

        print("--- Compilation du graphe LangGraph ---")
        memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=memory)
        print("--- Graphe compilé avec succès ---")

    def _grade_generation(self, state: GraphState) -> str:
        print("---DÉCISION: VÉRIFICATION DES HALLUCINATIONS ET DE LA PERTINENCE---")
        count = state.get("generation_count", 0)
        if count > 3:
            print("❌ Nombre maximum de tentatives de génération atteint. Échec.")
            return "fail"

        question, documents, generation = state["question"], state["documents"], state["generation"]
        
        hallucination_score = hallucination_grader.invoke({"documents": documents, "generation": generation})
        if not hallucination_score.binary_score:
            print(f"⚠️ La génération n'est pas basée sur les documents (Tentative {count}). Nouvelle tentative...")
            return "not supported"
        
        print("✅ La génération est basée sur les documents.")
        
        answer_score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_score.binary_score:
            print("✅ La génération est utile.")
            return "useful"
        else:
            print(f"⚠️ La génération n'est pas utile (Tentative {count}). Réécriture de la requête...")
            return "not useful"

    def _route_question(self, state: GraphState) -> str:
        print("---DÉCISION: ROUTAGE DE LA QUESTION---")
        if self.retriever:
            print("▶️ Retriever trouvé, routage vers RAG.")
            return RETRIEVE
        print("▶️ Pas de retriever, routage vers recherche web.")
        return WEBSEARCH

    def _decide_to_generate_or_rewrite(self, state: GraphState) -> str:
        print("---DÉCISION: GÉNÉRER OU RÉÉCRIRE---")
        count = state.get("query_rewrite_count", 0)
        
        if not state["documents"]:
            if count >= 3:
                print("❌ Nombre maximum de réécritures atteint. Basculement vers recherche web.")
                return WEBSEARCH
            else:
                print(f"⚠️ Aucun document pertinent (Tentative {count}). Réécriture de la requête...")
                return QUERY_REWRITE
        else:
            print(f"✅ {len(state['documents'])} documents pertinents trouvés. Poursuite vers la génération.")
            return GENERATE

    def ask_question(self, question: str, retriever: Optional[Any] = None, config: Optional[Dict] = None) -> Dict[str, Any]:
        if not self.app:
            raise RuntimeError("Le système RAG n'est pas initialisé.")

        self.retriever = retriever
        initial_state = {
            "question": question, "generation": "", "documents": [],
            "query_rewrite_count": 0, "generation_count": 0
        }

        print(f"\n--- Lancement du workflow pour la question : '{question}' ---")
        start_time = time.time()
        result = self.app.invoke(initial_state, config=config)
        end_time = time.time()
        print("--- Fin du workflow ---")
        self.retriever = None

        return {
            "success": True, "answer": result.get("generation", "Aucune réponse générée."),
            "question": question, "processing_time": round(end_time - start_time, 2),
            "documents_used": len(result.get("documents", [])),
            "query_rewrites": result.get("query_rewrite_count", 0),
            "documents": result.get("documents", []),
        }

# Instance unique (singleton) qui sera importée par Streamlit
rag_system = AdaptiveRAGSystem()
