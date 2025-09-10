"""
Modular RAG System for easy integration with Streamlit and other applications
"""
import time
import traceback
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

# Import your custom chains and nodes
from chains.answer_grader import answer_grader
from chains.hallucination_grader import hallucination_grader
from chains.retriever_grader import retrieval_grader
from chains.router_query import question_router
from Node_constant import (GRADE_DOCUMENTS, GENERATE, QUERY_REWRITE, RETRIEVE,
                           WEBSEARCH)
from nodes.generate import generate
from nodes.query_rewrite import query_rewrite
from nodes.web_search import web_search

# Load environment variables
load_dotenv()


# --- 1. CORRECTED GRAPH STATE DEFINITION ---
# The 'retriever' has been REMOVED from the state.
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question.
        generation: The LLM's generated answer.
        documents: A list of retrieved documents.
        query_rewrite_count: A counter for query rewrite attempts.
        generation_count: A counter for generation attempts.
    """
    question: str
    generation: str
    documents: List[Any]
    query_rewrite_count: int
    generation_count: int


class AdaptiveRAGSystem:
    """
    Modular Adaptive RAG System with Self-Reflection
    """

    def __init__(self):
        """Initialize the RAG system"""
        self.app = None
        self.retriever = None  # Initialize retriever attribute
        self._setup_workflow()

    # --- 2. CORRECTED RETRIEVE DOCUMENTS NODE ---
    # It now correctly defines 'question' before using it.
    def _retrieve_documents(self, state: GraphState) -> Dict[str, Any]:
        """
        Retrieves documents using the retriever attached to the class instance.
        """
        print("---RETRIEVE DOCUMENTS---")
        # CRITICAL FIX: Define 'question' from the state.
        question = state["question"]
        
        # Get retriever from the class instance, not the state.
        retriever = self.retriever

        documents = []
        if retriever is not None:
            print("📁 Using uploaded documents retriever")
            try:
                # 'question' is now safely defined before this call.
                documents = retriever.invoke(question)
                print(f"✅ Retrieved {len(documents)} documents from uploaded files")
            except Exception as e:
                print(f"❌ Error retrieving from uploaded documents: {e}")
                traceback.print_exc()
        else:
            print("⚠️ No custom retriever provided. Returning empty list.")
            
        return {"documents": documents, "question": question}

    def _grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """
        Determines whether the retrieved documents are relevant to the question.
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        for d in documents:
            try:
                score = retrieval_grader.invoke(
                    {"question": question, "document": d.page_content}
                )
                grade = score.binary_score
                
                if grade:
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
            except Exception:
                print(f"---GRADE: ERROR PROCESSING DOCUMENT, SKIPPING---")
                continue
                
        return {"documents": filtered_docs}

    def _setup_workflow(self):
        """Set up the LangGraph workflow"""
        self.workflow = StateGraph(GraphState)

        # Add nodes
        self.workflow.add_node(RETRIEVE, self._retrieve_documents)
        self.workflow.add_node(GRADE_DOCUMENTS, self._grade_documents)
        self.workflow.add_node(GENERATE, generate)
        self.workflow.add_node(WEBSEARCH, web_search)
        self.workflow.add_node(QUERY_REWRITE, query_rewrite)

        # Define edges and entry point
        self.workflow.set_conditional_entry_point(
            self._route_question,
            {WEBSEARCH: WEBSEARCH, RETRIEVE: RETRIEVE},
        )
        self.workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
        self.workflow.add_edge(QUERY_REWRITE, RETRIEVE)
        self.workflow.add_edge(WEBSEARCH, GENERATE)
        self.workflow.add_conditional_edges(
            GRADE_DOCUMENTS,
            self._decide_to_rewrite_query,
            {QUERY_REWRITE: QUERY_REWRITE, GENERATE: GENERATE, WEBSEARCH: WEBSEARCH},
        )
        self.workflow.add_conditional_edges(
            GENERATE,
            self._grade_generation,
            {"not supported": GENERATE, "useful": END, "not useful": QUERY_REWRITE, "fail": END},
        )

        # Compile the graph with memory
        print("--- Compiling LangGraph workflow ---")
        memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=memory)
        print("--- Workflow compiled successfully ---")

    def _grade_generation(self, state: GraphState) -> str:
        """Grades the generation for hallucinations and relevance."""
        print("---CHECK HALLUCINATIONS AND RELEVANCE---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        generation_count = state.get("generation_count", 0)

        # Hallucination check
        hallucination_score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        if not hallucination_score.binary_score:
            print("---DECISION: GENERATION IS NOT GROUNDED, RE-TRY---")
            return "not supported" if generation_count < 3 else "fail"
        
        print("---DECISION: GENERATION IS GROUNDED---")
        
        # Answer relevance check
        answer_score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_score.binary_score:
            print("---DECISION: GENERATION IS USEFUL---")
            return "useful"
        else:
            print("---DECISION: GENERATION IS NOT USEFUL, REWRITE QUERY---")
            return "not useful"

    def _route_question(self, state: GraphState) -> str:
        """Routes the question to web search or RAG based on retriever availability."""
        print("---ROUTE QUESTION---")
        if self.retriever is not None:
            print("---Custom retriever found. ROUTING TO RAG---")
            return RETRIEVE

        print("---No custom retriever. Using LLM router.---")
        source = question_router.invoke({"question": state["question"]})
        return WEBSEARCH if source.datasource == "web_search" else RETRIEVE

    def _decide_to_rewrite_query(self, state: GraphState) -> str:
        """Decides whether to rewrite the query, generate, or try a web search."""
        print("---ASSESSING DOCUMENT RELEVANCE---")
        if not state["documents"]:
            if state.get("query_rewrite_count", 0) >= 2: # Use >= for safety
                print("---DECISION: REWRITE FAILED. FALLING BACK TO WEB SEARCH.---")
                return WEBSEARCH
            else:
                print("---DECISION: NO RELEVANT DOCUMENTS. REWRITING QUERY.---")
                return QUERY_REWRITE
        else:
            print(f"---DECISION: {len(state['documents'])} RELEVANT DOCUMENTS. PROCEEDING TO GENERATE.---")
            return GENERATE

    # --- 3. CORRECTED ASK_QUESTION METHOD ---
    # It now correctly handles the retriever and the state.
    def ask_question(
        self,
        question: str,
        retriever: Optional[Any] = None,
        config: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Ask a question and get an answer from the RAG system"""
        if not self.app:
            raise Exception("RAG system not initialized")

        # Attach the retriever to the instance for this call ONLY.
        self.retriever = retriever

        # Initial state does NOT contain the retriever.
        initial_state = {
            "question": question,
            "generation": "",
            "documents": [],
            "query_rewrite_count": 0,
            "generation_count": 0,
        }

        print(f"--- Invoking RAG workflow for question: '{question}' ---")
        start_time = time.time()

        result = self.app.invoke(initial_state, config=config)

        # Clean up the retriever after the call.
        self.retriever = None
        end_time = time.time()
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
rag_system_instance = AdaptiveRAGSystem()


# Convenience function to be called from other modules like streamlit_app.py
'''def ask_question(
    question: str, retriever: Optional[Any] = None, config: Optional[Dict] = None
) -> Dict[str, Any]:
    """Convenience function to ask a question"""
    return rag_system_instance.ask_question(
        question, retriever=retriever, config=config
    )'''
