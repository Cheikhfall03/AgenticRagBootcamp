"""
Retriever node for document retrieval from uploaded documents or default knowledge base.

Note: This node should not be used directly in the graph since retriever management
is now handled at the system level to avoid serialization issues.
If you need to use this, you'll need to modify it to access the retriever
from the system instance rather than from state.
"""
from state import GraphState
from typing import List

def retrieve(state: GraphState):
    """
    WARNING: This function tries to get retriever from state, which will cause
    serialization issues with checkpointing. Use the system-level retriever instead.
    
    This is kept for compatibility but should be replaced with system-level
    retriever management.
    """
    print("---RETRIEVE DOCUMENTS---")
    print("⚠️ WARNING: This retriever node tries to access retriever from state.")
    print("⚠️ This may cause serialization issues. Consider using system-level retriever.")
    
    question = state["question"]
    # This will likely be None since retriever is no longer in state
    retriever = state.get("retriever")
    
    if retriever is not None:
        print("📁 Using retriever from state (not recommended)")
        try:
            documents = retriever.invoke(question)
            print(f"✅ Retrieved {len(documents)} documents")
            return {"documents": documents}
        except Exception as e:
            print(f"❌ Error retrieving documents: {e}")
            return {"documents": []}
    else:
        print("⚠️ No retriever found in state. Returning empty list.")
        return {"documents": []}
