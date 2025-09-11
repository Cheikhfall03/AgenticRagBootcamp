# nodes/retriever.py

from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from state import GraphState

def retrieve_documents(state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
    print("---NŒUD: RÉCUPÉRATION DE DOCUMENTS---")
    question = state["question"]

    retriever = config["configurable"].get("retriever")
    if retriever is None:
        print("⚠️ Aucun retriever fourni. Aucun document ne sera récupéré.")
        return {"documents": []}

    try:
        print(f"🔎 Utilisation du retriever: {type(retriever)}")
        documents = retriever.invoke(question)
        print(f"✅ {len(documents)} document(s) récupéré(s).")
        return {"documents": documents}
    except Exception as e:
        print(f"❌ Erreur lors de la récupération: {e}")
        return {"documents": []}
