"""
Retriever node for document retrieval from uploaded documents or default knowledge base.

This node is designed to work with the system-level retriever management
to avoid serialization issues with checkpointing.
"""
from state import GraphState
from typing import List, Dict, Any
import streamlit as st

def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve documents using Streamlit session state retriever.
    """
    print("---RETRIEVE DOCUMENTS FROM STREAMLIT SESSION---")
    
    question = state["question"]
    
    # Try to get retriever from Streamlit session state
    try:
        if 'retriever' in st.session_state:
            retriever = st.session_state.retriever
            print(f"✅ Found retriever in session state: {type(retriever)}")
            
            documents = retriever.invoke(question)
            print(f"✅ Retrieved {len(documents)} documents")
            return {"documents": documents}
        else:
            print("⚠️ No retriever found in session state")
            return {"documents": []}
            
    except Exception as e:
        print(f"❌ Error retrieving documents: {e}")
        return {"documents": []}


# Alternative: Global retriever approach (if session state doesn't work)
_global_retriever = None

def set_global_retriever(retriever):
    """Set a global retriever for standalone node usage"""
    global _global_retriever
    _global_retriever = retriever
    print(f"✅ Global retriever set: {type(retriever)}")

def retrieve_with_global(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve documents using the global retriever.
    """
    print("---RETRIEVE DOCUMENTS (GLOBAL RETRIEVER)---")
    
    question = state["question"]
    
    if _global_retriever is not None:
        print("📁 Using global retriever")
        try:
            documents = _global_retriever.invoke(question)
            print(f"✅ Retrieved {len(documents)} documents")
            return {"documents": documents}
        except Exception as e:
            print(f"❌ Error retrieving documents: {e}")
            return {"documents": []}
    else:
        print("⚠️ No global retriever set. Returning empty list.")
        return {"documents": []}
