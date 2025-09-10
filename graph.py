"""
Modular RAG System for easy integration with Streamlit and other applications
"""

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
# Correctly import all necessary chains, including the relevance_grader
from chains.answer_grader import answer_grader
from chains.retriever_grader import retrieval_grader
from Node_constant import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH, QUERY_REWRITE
from nodes.generate import generate
from chains.router_query import question_router
from chains.hallucination_grader import hallucination_grader
from nodes.web_search import web_search
from nodes.query_rewrite import query_rewrite
from typing import Dict, Any, Optional, List
import time
import traceback
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver # Ajout de l'importation pour MemorySaver

# Load environment variables
load_dotenv()

# --- SELF-CONTAINED STATE AND NODES ---

class GraphState(TypedDict):
    """
    Represents the state of our graph. All definitions are now in this file
    to prevent import issues.
    """
    question: str
    generation: str
    documents: List[Any]
    query_rewrite_count: int
    generation_count: int
    retriever: Optional[Any] # The custom retriever object

class AdaptiveRAGSystem:
    """
    Modular Adaptive RAG System with Self-Reflection
    """
    
    def __init__(self):
        """Initialize the RAG system"""
        self.app = None
        self._setup_workflow()
    
    def _retrieve_documents(self, state: GraphState) -> Dict[str, Any]:
        """
        Self-contained retrieve node logic.
        """
        print("---RETRIEVE DOCUMENTS---")
        question = state["question"]
        retriever = state.get("retriever")
        
        documents = []
        if retriever is not None:
            print("📁 Using uploaded documents retriever")
            try:
                documents = retriever.invoke(question)
                print(f"✅ Retrieved {len(documents)} documents from uploaded files")
            except Exception as e:
                print(f"❌ Error retrieving from uploaded documents: {e}")
        else:
            print("⚠️ No custom retriever provided. Returning empty list.")
            
        return {"documents": documents}

    def _grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """
        Self-contained grade_documents node logic with the correct boolean check.
        Determines whether the retrieved documents are relevant to the question.
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            # The grader returns a Pydantic model with a boolean 'binary_score'
            grade = score.binary_score
            
            # --- THE FIX IS HERE ---
            # Instead of checking for a "yes" string, we check the boolean directly.
            if grade:
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
        
        return {"documents": filtered_docs}

    def _setup_workflow(self):
        """Set up the LangGraph workflow"""
        self.workflow = StateGraph(GraphState)
        
        # Add nodes using internal methods where needed
        self.workflow.add_node(RETRIEVE, self._retrieve_documents)
        # Use the corrected, self-contained grading method
        self.workflow.add_node(GRADE_DOCUMENTS, self._grade_documents)
        self.workflow.add_node(GENERATE, generate)
        self.workflow.add_node(WEBSEARCH, web_search)
        self.workflow.add_node(QUERY_REWRITE, query_rewrite)
        
        self.workflow.set_conditional_entry_point(
            self._route_question,
            { WEBSEARCH: WEBSEARCH, RETRIEVE: RETRIEVE },
        )
        
        self.workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
        self.workflow.add_edge(QUERY_REWRITE, RETRIEVE)
        self.workflow.add_edge(WEBSEARCH, GENERATE)
        
        self.workflow.add_conditional_edges(
            GRADE_DOCUMENTS, self._decide_to_rewrite_query,
            { QUERY_REWRITE: QUERY_REWRITE, GENERATE: GENERATE, WEBSEARCH: WEBSEARCH },
        )
        
        self.workflow.add_conditional_edges(
            GENERATE, self._grade_generation_grounded_in_documents_and_question,
            { "not supported": GENERATE, "useful": END, "not useful": QUERY_REWRITE, "fail": END },
        )
        
        print("--- Compiling LangGraph workflow ---")
        memory = MemorySaver() # Création de l'instance de MemorySaver
        self.app = self.workflow.compile(checkpointer=memory) # Ajout du checkpointer à la compilation
        print("--- Workflow compiled successfully ---")

    def _grade_generation_grounded_in_documents_and_question(self, state: GraphState) -> str:
        """Grades the generation for hallucinations and relevance"""
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        generation_count = state.get("generation_count", 0)

        score = hallucination_grader.invoke({"documents": documents, "generation": generation})

        if not score.binary_score:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported" if generation_count < 3 else "fail"

        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        return "useful" if score.binary_score else "not useful"

    def _route_question(self, state: GraphState) -> str:
        print("---ROUTE QUESTION---")
        if state.get("retriever") is not None:
            print("---Custom retriever found. ROUTING TO RAG---")
            return RETRIEVE

        print("---No custom retriever. Using LLM router.---")
        source = question_router.invoke({"question": state["question"]})
        return WEBSEARCH if source.datasource == "web_search" else RETRIEVE

    def _decide_to_rewrite_query(self, state: GraphState) -> str:
        """Decides whether to rewrite the query, generate an answer, or try a web search."""
        print("---ASSESSING DOCUMENT RELEVANCE---")
        if not state["documents"]:
            if state.get("query_rewrite_count", 0) > 2:
                print("---DECISION: REWRITE FAILED. FALLING BACK TO WEB SEARCH.---")
                return WEBSEARCH
            else:
                print("---DECISION: NO RELEVANT DOCUMENTS FOUND. REWRITING QUERY.---")
                return QUERY_REWRITE
        else:
            print(f"---DECISION: {len(state['documents'])} RELEVANT DOCUMENTS FOUND. PROCEEDING TO GENERATE.---")
            return GENERATE

    # <-- MODIFIÉ : Ajout de 'config: Optional[Dict] = None' à la signature de la fonction
    def ask_question(self, question: str, retriever: Optional[Any] = None, config: Optional[Dict] = None) -> Dict[str, Any]:
        """Ask a question and get an answer from the RAG system"""
        if not self.app:
            raise Exception("RAG system not initialized")
        
        initial_state = {
            "question": question,
            "generation": "",
            "documents": [],
            "query_rewrite_count": 0,
            "generation_count": 0,
            "retriever": retriever
        }
        
        print(f"--- Invoking RAG workflow for question: {question} ---")
        start_time = time.time()
        
        # <-- MODIFIÉ : Utilisation du paramètre 'config' passé à la fonction
        result = self.app.invoke(initial_state, config=config)
        
        end_time = time.time()
        # Le print a été déplacé pour être plus logique
        print("--- Workflow invocation completed ---") 
        
        return {
            "success": True,
            "answer": result.get("generation", "No answer generated"),
            "question": question,
            "processing_time": round(end_time - start_time, 2),
            "documents_used": len(result.get("documents", [])),
            "query_rewrites": result.get("query_rewrite_count", 0),
            "documents": result.get("documents", []),
        }

# Singleton instance for easy import
rag_system = AdaptiveRAGSystem()

# <-- MODIFIÉ : Ajout de 'config: Optional[Dict] = None' pour relayer le paramètre
def ask_questions(question: str, retriever: Optional[Any] = None, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Convenience function to ask a question"""
    # <-- MODIFIÉ : Passage de 'config' à la méthode de la classe
    return rag_system.ask_question(question, retriever=retriever, config=config)
