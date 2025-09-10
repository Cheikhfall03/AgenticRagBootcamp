"""
Modular RAG System for easy integration with Streamlit and other applications
"""

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
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
from langgraph.checkpoint.memory import MemorySaver
from state import GraphState

# Load environment variables
load_dotenv()

class AdaptiveRAGSystem:
    """
    Modular Adaptive RAG System with Self-Reflection
    """
    
    def __init__(self):
        """Initialize the RAG system"""
        self.app = None
        self.current_retriever = None  # Store retriever outside of state
        self._setup_workflow()
    
    def _retrieve_documents(self, state: GraphState) -> Dict[str, Any]:
        """
        Retrieve documents using the current retriever.
        This replaces the external retrieve node to ensure retriever access.
        """
        print("---RETRIEVE DOCUMENTS---")
        question = state["question"]
        
        documents = []
        if self.current_retriever is not None:
            print("📁 Using uploaded documents retriever")
            try:
                documents = self.current_retriever.invoke(question)
                print(f"✅ Retrieved {len(documents)} documents from uploaded files")
            except Exception as e:
                print(f"❌ Error retrieving from uploaded documents: {e}")
                documents = []
        else:
            print("⚠️ No custom retriever provided. Returning empty list.")
            documents = []
            
        return {"documents": documents}

    def _grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """
        Grade documents for relevance to the question.
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        
        if not documents:
            print("---NO DOCUMENTS TO GRADE---")
            return {"documents": []}
        
        filtered_docs = []
        for d in documents:
            try:
                score = retrieval_grader.invoke(
                    {"question": question, "document": d.page_content}
                )
                # The grader returns a Pydantic model with a boolean 'binary_score'
                grade = score.binary_score
                
                if grade:
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
            except Exception as e:
                print(f"---ERROR GRADING DOCUMENT: {e}---")
                # If grading fails, include the document to be safe
                filtered_docs.append(d)
        
        print(f"---FILTERED TO {len(filtered_docs)} RELEVANT DOCUMENTS---")
        return {"documents": filtered_docs}

    def _setup_workflow(self):
        """Set up the LangGraph workflow"""
        self.workflow = StateGraph(GraphState)
        
        # Add nodes - using internal methods to ensure retriever access
        self.workflow.add_node(RETRIEVE, self._retrieve_documents)
        self.workflow.add_node(GRADE_DOCUMENTS, self._grade_documents)
        self.workflow.add_node(GENERATE, generate)
        self.workflow.add_node(WEBSEARCH, web_search)
        self.workflow.add_node(QUERY_REWRITE, query_rewrite)
        
        # Set conditional entry point
        self.workflow.set_conditional_entry_point(
            self._route_question,
            {
                WEBSEARCH: WEBSEARCH, 
                RETRIEVE: RETRIEVE
            },
        )
        
        # Add edges
        self.workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
        self.workflow.add_edge(QUERY_REWRITE, RETRIEVE)
        self.workflow.add_edge(WEBSEARCH, GENERATE)
        
        # Add conditional edges
        self.workflow.add_conditional_edges(
            GRADE_DOCUMENTS, 
            self._decide_to_rewrite_query,
            {
                QUERY_REWRITE: QUERY_REWRITE, 
                GENERATE: GENERATE, 
                WEBSEARCH: WEBSEARCH
            },
        )
        
        self.workflow.add_conditional_edges(
            GENERATE, 
            self._grade_generation_grounded_in_documents_and_question,
            {
                "not supported": GENERATE, 
                "useful": END, 
                "not useful": QUERY_REWRITE, 
                "fail": END
            },
        )
        
        print("--- Compiling LangGraph workflow ---")
        memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=memory)
        print("--- Workflow compiled successfully ---")

    def _grade_generation_grounded_in_documents_and_question(self, state: GraphState) -> str:
        """Grades the generation for hallucinations and relevance"""
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        generation_count = state.get("generation_count", 0)

        try:
            score = hallucination_grader.invoke({"documents": documents, "generation": generation})

            if not score.binary_score:
                print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
                return "not supported" if generation_count < 3 else "fail"

            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            score = answer_grader.invoke({"question": question, "generation": generation})
            return "useful" if score.binary_score else "not useful"
            
        except Exception as e:
            print(f"---ERROR IN GRADING: {e}---")
            return "fail"

    def _route_question(self, state: GraphState) -> str:
        """Route the question to appropriate processing path"""
        print("---ROUTE QUESTION---")
        
        # Check if we have a retriever set in the instance
        if self.current_retriever is not None:
            print("---Custom retriever found. ROUTING TO RAG---")
            return RETRIEVE

        print("---No custom retriever. Using LLM router.---")
        try:
            source = question_router.invoke({"question": state["question"]})
            routing_decision = WEBSEARCH if source.datasource == "web_search" else RETRIEVE
            print(f"---LLM ROUTER DECISION: {routing_decision}---")
            return routing_decision
        except Exception as e:
            print(f"---ERROR IN ROUTING: {e}. DEFAULTING TO WEBSEARCH---")
            return WEBSEARCH

    def _decide_to_rewrite_query(self, state: GraphState) -> str:
        """Decides whether to rewrite the query, generate an answer, or try a web search."""
        print("---ASSESSING DOCUMENT RELEVANCE---")
        
        documents = state.get("documents", [])
        query_rewrite_count = state.get("query_rewrite_count", 0)
        
        if not documents:
            if query_rewrite_count >= 2:
                print("---DECISION: REWRITE LIMIT REACHED. FALLING BACK TO WEB SEARCH.---")
                return WEBSEARCH
            else:
                print("---DECISION: NO RELEVANT DOCUMENTS FOUND. REWRITING QUERY.---")
                return QUERY_REWRITE
        else:
            print(f"---DECISION: {len(documents)} RELEVANT DOCUMENTS FOUND. PROCEEDING TO GENERATE.---")
            return GENERATE

    def ask_question(self, question: str, retriever: Optional[Any] = None, config: Optional[Dict] = None) -> Dict[str, Any]:
        """Ask a question and get an answer from the RAG system"""
        if not self.app:
            raise Exception("RAG system not initialized")
        
        # CRITICAL: Set the retriever in the instance BEFORE invoking
        self.current_retriever = retriever
        
        if retriever:
            print(f"✅ Retriever set: {type(retriever)}")
        else:
            print("⚠️ No retriever provided - will use web search if needed")
        
        initial_state = {
            "question": question,
            "generation": "",
            "documents": [],
            "file_paths": [],
            "web_search": False,
            "query_rewrite_count": 0,
            "generation_count": 0,
        }
        
        print(f"--- Invoking RAG workflow for question: {question} ---")
        start_time = time.time()
        
        try:
            result = self.app.invoke(initial_state, config=config)
            end_time = time.time()
            print("--- Workflow invocation completed successfully ---")
            
            return {
                "success": True,
                "answer": result.get("generation", "No answer generated"),
                "question": question,
                "processing_time": round(end_time - start_time, 2),
                "documents_used": len(result.get("documents", [])),
                "query_rewrites": result.get("query_rewrite_count", 0),
                "documents": result.get("documents", []),
            }
            
        except Exception as e:
            end_time = time.time()
            print(f"--- Workflow invocation failed: {e} ---")
            traceback.print_exc()
            
            return {
                "success": False,
                "answer": f"Error processing question: {str(e)}",
                "question": question,
                "processing_time": round(end_time - start_time, 2),
                "documents_used": 0,
                "query_rewrites": 0,
                "documents": [],
            }

# Singleton instance for easy import
rag_system = AdaptiveRAGSystem()

def ask_question(question: str, retriever: Optional[Any] = None, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Convenience function to ask a question"""
    return rag_system.ask_question(question, retriever=retriever, config=config)
