"""
Retriever node for document retrieval from uploaded documents or default knowledge base.

This node is designed to work with the system-level retriever management
to avoid serialization issues with checkpointing.
"""
from state import GraphState
from typing import List, Dict, Any

def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve documents. 
    
    Note: This function expects the retriever to be managed at the system level.
    If you're using this node directly in a graph, you'll need to ensure
    the retriever is accessible through some other mechanism.
    """
    print("---RETRIEVE DOCUMENTS (STANDALONE NODE)---")
    print("⚠️ WARNING: This node expects system-level retriever management.")
    
    question = state.get("question", "")
    
    # This standalone node doesn't have access to the system retriever
    # It would need to be modified to work with your specific setup
    print("⚠️ No retriever access in standalone mode. Returning empty list.")
    print("⚠️ Use the system-level _retrieve_documents method instead.")
    
    return {"documents": []}


# Alternative function that could work with a global retriever if needed
_global_retriever = None

def set_global_retriever(retriever):
    """Set a global retriever for standalone node usage"""
    global _global_retriever
    _global_retriever = retriever
    print(f"✅ Global retriever set: {type(retriever)}")

def _call_global_retriever(retriever, question: str):
    """Same robust call pattern as in graph.py"""
    if retriever is None:
        return []
    for method_name in ("get_relevant_documents", "get_relevant_texts", "retrieve", "invoke"):
        method = getattr(retriever, method_name, None)
        if callable(method):
            try:
                result = method(question)
                return result if result is not None else []
            except Exception as e:
                print(f"⚠️ Global retriever method '{method_name}' raised: {e}")
                continue
    if callable(retriever):
        try:
            return retriever(question)
        except Exception as e:
            print(f"⚠️ Calling global retriever as callable failed: {e}")
    print("⚠️ No usable method found on global retriever.")
    return []

def retrieve_with_global(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve documents using the global retriever.
    This is an alternative approach if you need standalone node functionality.
    """
    print("---RETRIEVE DOCUMENTS (GLOBAL RETRIEVER)---")
    
    question = state.get("question", "")
    
    if _global_retriever is not None:
        print("📁 Using global retriever")
        try:
            documents = _call_global_retriever(_global_retriever, question)
            documents = list(documents) if documents is not None else []
            print(f"✅ Retrieved {len(documents)} documents")
            return {"documents": documents}
        except Exception as e:
            print(f"❌ Error retrieving documents: {e}")
            return {"documents": []}
    else:
        print("⚠️ No global retriever set. Returning empty list.")
        return {"documents": []}
