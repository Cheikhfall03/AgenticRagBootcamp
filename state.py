from typing import List, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question.
        generation: The LLM's generated answer.
        documents: A list of retrieved documents.
        chat_history: The history of the conversation.
        file_paths: Paths to any user-uploaded files for the current query.
        web_search: A flag indicating if a web search is needed.
        query_rewrite_count: A counter for query rewrite attempts.
        generation_count: A counter for generation attempts (for hallucination retries).
    """
    question: str
    generation: str
    documents: List[Any]  # Can be Document objects or strings
    file_paths: List[str]
    web_search: bool
    query_rewrite_count: int
    generation_count: int