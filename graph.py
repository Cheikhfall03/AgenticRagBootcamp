import time
import traceback
from typing import Dict, Any, Optional, Iterator
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

    def _retrieve_documents(self, state: GraphState) -> Dict[str, Any]:
        """
        Récupère les documents en utilisant le retriever actuellement actif (soit par défaut, soit personnalisé).
        Ce nœud est maintenant interne à la classe et fiable.
        """
        print("---NŒUD: RÉCUPÉRATION DE DOCUMENTS---")
        question = state["question"]

        if self.current_retriever is None:
            print("⚠️ Aucun retriever n'est disponible. Retourne une liste vide.")
            return {"documents": []}

        try:
            print(f"🔎 Utilisation du retriever: {type(self.current_retriever)}")
            documents = self.current_retriever.invoke(question)
            print(f"✅ {len(documents)} document(s) récupéré(s).")
            return {"documents": documents}
        except Exception as e:
            print(f"❌ Erreur lors de la récupération de documents: {e}")
            return {"documents": []}

    def _grade_documents(self, state: GraphState) -> Dict[str, Any]:
        print("---NŒUD: ÉVALUATION DE LA PERTINENCE DES DOCUMENTS---")
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
                print(f"⚠️ Erreur d'évaluation, inclusion par défaut: {e}")
                filtered_docs.append(d) # En cas d'erreur, on garde le doc par sécurité

        return {"documents": filtered_docs}

    def _route_question(self, state: GraphState) -> str:
        print("---ROUTE QUESTION---")
        question = state["question"]
        source: RouteQuery = question_router.invoke({"question": question})
        if source.datasource == WEBSEARCH:
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return WEBSEARCH
        elif source.datasource == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return RETRIEVE

    def _decide_to_generate(self, state: GraphState) -> str:
        """
        Décide s'il faut générer une réponse, réécrire la question, ou passer à la recherche web.
        """
        print("---NŒUD: DÉCISION POST-RÉCUPÉRATION---")
        if state["documents"]:
            print("✅ Documents pertinents trouvés. Passage à la génération.")
            return GENERATE
        else:
            if state["query_rewrite_count"] < 1: # On autorise 1 réécriture
                print("⛔ Aucun document pertinent. Tentative de réécriture de la question.")
                return QUERY_REWRITE
            else:
                print("⛔ Échec après réécriture. Passage à la recherche web comme dernier recours.")
                return WEBSEARCH

    def _grade_generation(self, state: GraphState) -> str:
        """Evaluates the generation for hallucinations and relevance."""
        print("---NODE: GRADE GENERATION---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        if not generation:
            print("⛔ Generation is empty. Ending.")
            return END

        # If the generation came from a web search, we can't do a hallucination check.
        # We just check if the answer is useful.
        if not documents:
            print("⚠️ No documents to check for hallucinations (likely from web search).")
            answer_score = answer_grader.invoke({"question": question, "generation": generation})
            if getattr(answer_score, "binary_score", False):
                print("✅ Web search answer deemed USEFUL.")
                return END
            else:
                print("⛔ Web search answer deemed NOT USEFUL.")
                return END

        # Combine and truncate the document context to prevent exceeding the model's token limit.
        MAX_CONTEXT_CHARS = 15000  # Approx. 3750 tokens, a safe limit for most models.
        docs_texts = [getattr(d, "page_content", str(d)) for d in documents]
        full_context = "\n\n---\n\n".join(docs_texts)

        if len(full_context) > MAX_CONTEXT_CHARS:
            print(f"⚠️ Context length ({len(full_context)} chars) is too long. Truncating to {MAX_CONTEXT_CHARS}.")
            full_context = full_context[:MAX_CONTEXT_CHARS]

        print("---CHECKING FOR HALLUCINATIONS---")
        if isinstance(full_context, list):
            documents_text = "\n\n".join([doc.page_content for doc in full_context])
        else:
            documents_text = str(full_context)

        try:
            hallucination_score = hallucination_grader.invoke({
                "documents": documents_text[:5000],  # 🔒 Sécurise la taille du contexte
                "generation": generation[:2000]      # 🔒 Limite aussi la génération
            })
        except Exception as e:
            print(f"⚠️ Erreur dans hallucination_grader: {e}")
            # On considère par défaut que c’est grounded pour ne pas bloquer
            return END

        if not getattr(hallucination_score, "binary_score", False):
            print("⛔ DECISION: HALLUCINATION DETECTED. Re-trying generation.")
            if state["generation_count"] < 1: # Allow one retry
                return GENERATE
            else:
                print("⛔ Failed to correct hallucination after retry. Ending.")
                return END

        print("✅ DECISION: GENERATION IS GROUNDED in documents.")
        print("---CHECKING FOR ANSWER RELEVANCE---")
        answer_score = answer_grader.invoke({"question": question, "generation": generation})
        if getattr(answer_score, "binary_score", False):
            print("✅ Answer is RELEVANT and GROUNDED. Ending.")
            return END
        else:
            print("⛔ Answer is NOT RELEVANT, but grounded. Ending.")
            return END

    def _setup_workflow(self):
        """Construit et connecte les nœuds du graphe LangGraph."""
        self.workflow.set_entry_point(RETRIEVE) # <--- POINT D'ENTRÉE SIMPLIFIÉ

        self.workflow.add_node(RETRIEVE, self._retrieve_documents)
        self.workflow.add_node(GRADE_DOCUMENTS, self._grade_documents)
        self.workflow.add_node(QUERY_REWRITE, query_rewrite)
        self.workflow.add_node(WEBSEARCH, web_search)
        self.workflow.add_node(GENERATE, generate)

        # --- Définition des chemins (edges) ---
        self.workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
        self.workflow.add_edge(QUERY_REWRITE, RETRIEVE)
        self.workflow.add_edge(WEBSEARCH, GENERATE)

        self.workflow.add_conditional_edges(
            GRADE_DOCUMENTS,
            self._decide_to_generate,
            {
                GENERATE: GENERATE,
                QUERY_REWRITE: QUERY_REWRITE,
                WEBSEARCH: WEBSEARCH
            }
        )
        self.workflow.add_conditional_edges(
            GENERATE,
            self._grade_generation,
            {
                GENERATE: GENERATE,
                END: END
            }
        )

    def run(self, question: str, retriever: Optional[Any] = None, config: Optional[Dict] = None) -> Iterator[Dict[str, Any]]:
        """
        Point d'entrée principal pour exécuter le graphe.
        Il sélectionne le bon retriever puis lance le flux (stream).
        """
        if not self.app:
            print("Erreur: Le système RAG n'est pas compilé.")
            return iter([]) # Return an empty iterator

        # Si un retriever est fourni (fichiers uploadés), on l'utilise.
        # Sinon, on utilise le retriever par défaut initialisé au démarrage.
        self.current_retriever = retriever if retriever is not None else self.default_retriever

        if self.current_retriever is None:
            print("Erreur: Aucun retriever n'est disponible (ni par défaut, ni personnalisé).")
            return iter([]) # Return an empty iterator

        initial_state = {
            "question": question,
            "query_rewrite_count": 0,
            "generation_count": 0,
            "documents": [],
            "generation": ""
        }

        print(f"--- Lancement du graphe pour la question: '{question}' ---")
        
        # Return the stream generator directly
        return self.app.stream(initial_state, config=config)

# --- Instance unique (Singleton) pour l'application ---
rag_system = AdaptiveRAGSystem()

