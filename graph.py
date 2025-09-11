# graph.py

from langchain_core.runnables import RunnableConfig
import time, traceback, os
from typing import Dict, Any, Optional, Iterator
from dotenv import load_dotenv
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
from nodes.retriever import retrieve_documents   # ✅ Nouveau import
from Node_constant import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH, QUERY_REWRITE, ROUTE_QUESTION
from state import GraphState

# --- Initialisation ---
load_dotenv()

class AdaptiveRAGSystem:
    def __init__(self):
        self.workflow = StateGraph(GraphState)
        self._setup_workflow()
        memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=memory)
        print("✅ Graphe LangGraph compilé avec succès.")

    def _route_question(self, state: GraphState) -> Dict[str, Any]:
        print("---NŒUD: ROUTAGE DE LA QUESTION---")
        question = state["question"]
        try:
            source: RouteQuery = question_router.invoke({"question": question})
            datasource = str(source.datasource).strip().lower()
            if datasource == WEBSEARCH:
                return {"next": WEBSEARCH}
            elif datasource == RETRIEVE:
                return {"next": RETRIEVE}
            else:
                return {"next": RETRIEVE}
        except Exception as e:
            print(f"⚠️ Erreur de routage: {e}")
            return {"next": RETRIEVE}

    def _grade_documents(self, state: GraphState) -> Dict[str, Any]:
        print("---NŒUD: ÉVALUATION DOCUMENTS---")
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
                    filtered_docs.append(d)
            except Exception as e:
                print(f"⚠️ Erreur d’éval: {e}")
                filtered_docs.append(d)
        return {"documents": filtered_docs, "question": question}

    def _decide_to_generate(self, state: GraphState) -> str:
        if state["documents"]:
            return GENERATE
        else:
            return QUERY_REWRITE if state["query_rewrite_count"] < 1 else WEBSEARCH

    def _grade_generation(self, state: GraphState) -> str:
        generation = state["generation"]
        if not generation:
            return END
        return END

    def _setup_workflow(self):
        self.workflow.add_node(RETRIEVE, retrieve_documents)  # ✅ Utilise le nœud importé
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
            return iter([])
        if config is None:
            config = {"configurable": {}}
        config["configurable"]["retriever"] = retriever
        initial_state = {
            "question": question,
            "query_rewrite_count": 0,
            "generation_count": 0,
            "documents": [],
            "generation": "",
            "file_paths": [],
            "web_search": False,
            "route": "",
        }
        return self.app.stream(initial_state, config=config)

# --- Singleton ---
rag_system = AdaptiveRAGSystem()
