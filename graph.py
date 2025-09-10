# graph.py (Version Adaptée)

import time
import traceback
from typing import Dict, Any, Optional, Iterator
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# --- Import des composants (inchangé) ---
from chains.answer_grader import answer_grader
from chains.retriever_grader import retrieval_grader
from chains.router_query import question_router
from chains.hallucination_grader import hallucination_grader
from nodes.generate import generate
from nodes.web_search import web_search
from nodes.query_rewrite import query_rewrite
from Node_constant import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH, QUERY_REWRITE
from state import GraphState
from ingestion.ingestion import initialize_default_retriever

load_dotenv()

class AdaptiveRAGSystem:
    """Système RAG modulaire, robuste et centralisé."""

    def __init__(self):
        """Initialise le système RAG avec un retriever par défaut."""
        self.workflow = StateGraph(GraphState)

        try:
            self.default_retriever = initialize_default_retriever()
            print("✅ Default retriever initialized.")
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Could not initialize default retriever: {e}")
            self.default_retriever = None

        # This will hold the retriever for the current execution run
        self.current_retriever = None

        self._setup_workflow()
        memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=memory)
        print("✅ LangGraph graph compiled successfully.")

    def _retrieve_documents(self, state: GraphState) -> Dict[str, Any]:
        """
        Nœud de récupération simple qui utilise le 'current_retriever'
        défini pour l'exécution en cours.
        """
        print("---NODE: RETRIEVE DOCUMENTS---")
        question = state["question"]

        if self.current_retriever is None:
            print("⚠️ No retriever available for this run. Returning empty list.")
            return {"documents": []}

        try:
            print(f"🔎 Using retriever: {type(self.current_retriever)}")
            documents = self.current_retriever.invoke(question)
            print(f"✅ Retrieved {len(documents)} document(s).")
            return {"documents": documents}
        except Exception as e:
            print(f"❌ Error during document retrieval: {e}")
            return {"documents": []}

    # ... (les autres nœuds comme _grade_documents, _decide_to_generate, etc. restent inchangés) ...
    def _grade_documents(self, state: GraphState) -> Dict[str, Any]:
        # ... (votre code existant ici)
        pass
    def _decide_to_generate(self, state: GraphState) -> str:
        # ... (votre code existant ici)
        pass
    def _grade_generation(self, state: GraphState) -> str:
        # ... (votre code existant ici)
        pass


    def _setup_workflow(self):
        """Construit et connecte les nœuds du graphe LangGraph."""
        self.workflow.set_entry_point(RETRIEVE)
        self.workflow.add_node(RETRIEVE, self._retrieve_documents)
        self.workflow.add_node(GRADE_DOCUMENTS, self._grade_documents)
        # ... (le reste de votre setup_workflow reste inchangé) ...


    def run(self, question: str, retriever: Optional[Any] = None, config: Optional[Dict] = None) -> Iterator[Dict[str, Any]]:
        """
        Point d'entrée principal et unique pour exécuter le graphe.
        Il sélectionne le bon retriever puis lance le flux (stream).
        """
        # Étape 1: Sélectionner le retriever pour cette exécution spécifique
        if retriever is not None:
            print("✅ Using user-provided retriever for this run.")
            self.current_retriever = retriever
        else:
            print("🌐 Using default knowledge base for this run.")
            self.current_retriever = self.default_retriever

        # Étape 2: Définir l'état initial
        initial_state = {
            "question": question,
            "generation": "",
            "documents": [],
            "query_rewrite_count": 0,
            "generation_count": 0
        }

        # Étape 3: Exécuter le graphe et retourner le générateur de flux
        print(f"--- Starting graph run for question: '{question}' ---")
        return self.app.stream(initial_state, config=config)

# --- Instance unique pour l'application ---
rag_system = AdaptiveRAGSystem()
