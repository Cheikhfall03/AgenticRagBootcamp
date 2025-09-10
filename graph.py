# graph.py (Version Corrigée)
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
from ingestion import initialize_default_retriever

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

    # ... (les autres fonctions du graphe comme _grade_documents, etc. restent les mêmes) ...

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
        """
        Route toujours vers la récupération de documents, car nous avons toujours un retriever.
        """
        print("---NŒUD: ROUTAGE DE LA QUESTION---")
        print("➡️ Toujours router vers RETRIEVE car un retriever (défaut ou custom) est toujours disponible.")
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
        """Évalue la génération pour les hallucinations et la pertinence."""
        print("---NŒUD: ÉVALUATION DE LA GÉNÉRATION---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        
        if not documents: # Si on vient du web search, on ne peut pas vérifier les hallucinations
            print("⚠️ Impossible de vérifier les hallucinations (source web). On vérifie la pertinence de la réponse.")
            score = answer_grader.invoke({"question": question, "generation": generation})
            if getattr(score, "binary_score", False):
                print("✅ Réponse jugée utile.")
                return END
            else:
                print("⛔ Réponse jugée non utile. Fin.")
                return END # On pourrait boucler, mais finissons ici pour éviter les boucles infinies.

        docs_texts = [getattr(d, "page_content", str(d)) for d in documents]
        hallucination_score = hallucination_grader.invoke({"documents": docs_texts, "generation": generation})
        
        if not getattr(hallucination_score, "binary_score", False):
            print("⛔ HALLUCINATION DÉTECTÉE ! Tentative de re-génération.")
            if state["generation_count"] < 1:
                return GENERATE # On re-génère une fois
            else:
                print("⛔ Échec de la re-génération. Fin.")
                return END
                
        print("✅ Aucune hallucination détectée.")
        answer_score = answer_grader.invoke({"question": question, "generation": generation})
        if getattr(answer_score, "binary_score", False):
            print("✅ Réponse jugée utile.")
            return END
        else:
            print("⛔ Réponse jugée non utile, mais non hallucinatoire. Fin.")
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
        
    def ask_question(self, question: str, retriever: Optional[Any] = None, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Point d'entrée principal pour poser une question au système.
        Gère intelligemment le retriever à utiliser.
        """
        if not self.app:
            return {"success": False, "answer": "Erreur: Le système RAG n'est pas compilé."}
            
        # --- LOGIQUE CENTRALE CORRIGÉE ---
        # Si un retriever est fourni (fichiers uploadés), on l'utilise.
        # Sinon, on utilise le retriever par défaut initialisé au démarrage.
        self.current_retriever = retriever if retriever is not None else self.default_retriever
        
        if self.current_retriever is None:
             return {"success": False, "answer": "Erreur: Aucun retriever n'est disponible (ni par défaut, ni personnalisé)."}

        initial_state = {"question": question, "query_rewrite_count": 0, "generation_count": 0}
        
        print(f"--- Lancement du graphe pour la question: '{question}' ---")
        start_time = time.time()
        
        final_state = None
        try:
            for event in self.app.stream(initial_state, config=config, stream_mode="values"):
                final_state = event
            
            end_time = time.time()
            
            answer = final_state.get("generation", "Aucune réponse n'a pu être générée.")
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
            print(f"--- ERREUR D'EXÉCUTION DU GRAPHE: {e} ---")
            traceback.print_exc()
            return {
                "success": False,
                "answer": f"Une erreur est survenue: {e}",
                "processing_time": round(end_time - start_time, 2),
            }

# --- Instance unique (Singleton) pour l'application ---
rag_system = AdaptiveRAGSystem()
