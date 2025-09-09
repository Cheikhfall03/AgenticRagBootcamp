"""
Retriever node for document retrieval from uploaded documents or default knowledge base.
"""
from state import GraphState
from typing import List


def retrieve(state: GraphState):
    """
    Retrieve documents using the retriever from state or default retriever
    """
    print("---RETRIEVE DOCUMENTS---")
    
    question = state["question"]
    retriever = state.get("retriever")
    
    # Check if we have a retriever from uploaded documents
    if retriever is not None:
        print("📁 Using uploaded documents retriever")
        try:
            documents = retriever.invoke(question)
            print(f"✅ Retrieved {len(documents)} documents from uploaded files")
            return {"documents": documents}
        except Exception as e:
            print(f"❌ Error retrieving from uploaded documents: {e}")
            return {"documents": []}
