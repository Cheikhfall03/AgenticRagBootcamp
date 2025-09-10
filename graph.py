# graph.py (Version Finalisée & Nettoyée)

import time
import traceback
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# --- Import des composants du graphe ---
from chains.answer_grader import answer_grader
from chains.retriever_grader import retrieval_grader
from chains.router_query import question_router
from chains.hallucination_grader import hallucination_grader
from nodes.generate import generate
from nodes.web_search import web_search
from nodes.query_rewrite import query_rewrite
from Node_constant import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH, QUERY_REWRITE
from state import GraphState

# --- Import de l'initialiseur du retriever par défaut ---
from ingestion.ingestion import initialize_default_retriever

load_dotenv()


class AdaptiveRAGSystem:
    """Système RAG modulaire, robuste et centralisé."""

    def __init__(self):
        """Initialise le système RAG avec un retriever par défaut."""
        self.workflow = StateGraph(GraphState)

        # --- Initialisation du retriever par défaut ---
        try:
            self.default_retriever = initialize_default_retriever()
        except Exception as e:
            print(f"❌ ERREUR CRITIQUE: Impossible d'initialiser le retriever par défaut: {e}")
            self.default_retriever = None

        self.current_retriever = self.default_retriever

        # --- Construction du graphe ---
        self._setup_workflow()

        # --- Compilation du graphe ---
        memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=memory)
        print("✅ Graphe LangGraph compilé avec succès.")

    # --- Nœud : récupération des documents ---
    def _retrieve_documents(self, state: GraphState) -> Dict[str, Any]:
        print("---NŒUD: RÉCUPÉRATION DE DOCUMENTS---")
        question = state["question"]

        if self.current_retriever is None:
            print("⚠️ Aucun retriever disponible. Retourne une liste vide.")
            return {"documents": []}

        try:
            print(f"🔎 Utilisation du retriever: {type(self.current_retriever)}")
            documents = self.current_retriever.invoke(question)
            print(f"✅ {len(documents)} document(s) récupéré(s).")
            return {"documents": documents}
        except Exception as e:
            print(f"❌ Erreur lors de la récupération: {e}")
            return {"documents": []}

    # --- Nœud : évaluation de pertinence des documents ---
    def _grade_documents(self, state: GraphState) -> Dict[str, Any]:
        print("---NŒUD: ÉVALUATION PERTINENCE DOCUMENTS---")
        question = state["question"]
        documents = state["documents"]

        if not documents:
            return {"documents": []}

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
                print(f"⚠️ Erreur d'évaluation → inclusion par défaut: {e}")
                filtered_docs.append(d)

        return {"documents": filtered_docs}

    # --- Nœud : routage de la question ---
    def _route_question(self, state: GraphState) -> str:
        print("---NŒUD: ROUTAGE QUESTION---")
        print("➡️ Toujours router vers RETRIEVE (retriever dispo).")
        return RETRIEVE

    # --- Nœud : décision après récupération ---
    def _decide_to_generate(self, state: GraphState) -> str:
        print("---NŒUD: DÉCISION POST-RÉCUPÉRATION---")
        if state["documents"]:
            print("✅ Documents trouvés → Génération.")
            return GENERATE
        if state["query_rewrite_count"] < 1:
            print("⛔ Aucun doc pertinent → Réécriture de la question.")
            return QUERY_REWRITE
        print("⛔ Échec après réécriture → Recherche Web.")
        return WEBSEARCH

    # --- Nœud : évaluation de la génération ---
    def _grade_generation(self, state: GraphState) -> str:
        print("---NŒUD: ÉVALUATION GÉNÉRATION---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        if not generation:
            print("⛔ Generation vide. Fin.")
            return END

        # Cas Web Search → pas de check hallucination
        if not documents:
            print("⚠️ Réponse issue du web → Vérification utilité.")
            answer_score = answer_grader.invoke({"question": question, "generation": generation})
            if getattr(answer_score, "binary_score", False):
                print("✅ Réponse Web UTILE.")
            else:
                print("⛔ Réponse Web NON UTILE.")
            return END

        # Préparation du contexte
        MAX_CONTEXT_CHARS = 15000
        docs_texts = [getattr(d, "page_content", str(d)) for d in documents]
        full_context = "\n\n---\n\n".join(docs_texts)

        if len(full_context) > MAX_CONTEXT_CHARS:
            print(f"⚠️ Contexte trop long ({len(full_context)}). Troncature.")
            full_context = full_context[:MAX_CONTEXT_CHARS]

        documents_text = str(full_context)

        # Vérification hallucinations
        try:
            hallucination_score = hallucination_grader.invoke({
                "documents": documents_text[:5000],
                "generation": generation[:2000]
            })
        except Exception as e:
            print(f"⚠️ Erreur hallucination_grader: {e}")
            return END

        if not getattr(hallucination_score, "binary_score", False):
            print("⛔ HALLUCINATION détectée → Retry si possible.")
            if state["generation_count"] < 1:
                return GENERATE
            print("⛔ Échec après retry. Fin.")
            return END

        # Vérification pertinence
        print("✅ Génération ancrée. Vérification pertinence.")
        answer_score = answer_grader.invoke({"question": question, "generation": generation})
        if getattr(answer_score, "binary_score", False):
            print("✅ Réponse PERTINENTE & ANCRÉE.")
        else:
            print("⛔ Réponse NON pertinente (mais ancrée).")
        return END

    # --- Construction du graphe ---
    def _setup_workflow(self):
        self.workflow.set_entry_point(RETRIEVE)

        self.workflow.add_node(RETRIEVE, self._retrieve_documents)
        self.workflow.add_node(GRADE_DOCUMENTS, self._grade_documents)
        self.workflow.add_node(QUERY_REWRITE, query_rewrite)
        self.workflow.add_node(WEBSEARCH, web_search)
        self.workflow.add_node(GENERATE, generate)

        # Edges
        self.workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
        self.workflow.add_edge(QUERY_REWRITE, RETRIEVE)
        self.workflow.add_edge(WEBSEARCH, GENERATE)

        self.workflow.add_conditional_edges(
            GRADE_DOCUMENTS,
            self._decide_to_generate,
            {GENERATE: GENERATE, QUERY_REWRITE: QUERY_REWRITE, WEBSEARCH: WEBSEARCH},
        )
        self.workflow.add_conditional_edges(
            GENERATE,
            self._grade_generation,
            {GENERATE: GENERATE, END: END},
        )

    # --- API publique ---
    def ask_question(self, question: str, retriever: Optional[Any] = None, config: Optional[Dict] = None) -> Dict[str, Any]:
        if not self.app:
            return {"success": False, "answer": "Erreur: Graphe RAG non compilé."}

        # Choix du retriever (custom ou défaut)
        self.current_retriever = retriever if retriever is not None else self.default_retriever
        if self.current_retriever is None:
            return {"success": False, "answer": "Erreur: Aucun retriever disponible."}

        initial_state = {"question": question, "query_rewrite_count": 0, "generation_count": 0}
        print(f"--- Lancement graphe pour question: '{question}' ---")
        start_time = time.time()

        final_state = None
        try:
            for event in self.app.stream(initial_state, config=config, stream_mode="values"):
                final_state = event

            end_time = time.time()
            answer = final_state.get("generation", "Aucune réponse générée.")
            docs_used = final_state.get("documents", [])

            return {
                "success": True,
                "answer": answer,
                "processing_time": round(end_time - start_time, 2),
                "documents_used": len(docs_used),
                "query_rewrites": final_state.get("query_rewrite_count", 0),
                "documents": docs_used,
            }
        except Exception as e:
            end_time = time.time()
            print(f"--- ERREUR EXECUTION GRAPHE: {e} ---")
            traceback.print_exc()
            return {
                "success": False,
                "answer": f"Erreur: {e}",
                "processing_time": round(end_time - start_time, 2),
            }


# --- Instance unique pour l'application ---
rag_system = AdaptiveRAGSystem()
