from state import GraphState
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

def generate(state: GraphState):
    """
    Generates an answer using the retrieved documents as context.
    It can handle a mixed list of Document objects and plain strings.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        dict: A dictionary containing the updated state.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation_count = state.get("generation_count", 0) + 1

    # Create the prompt template
    prompt_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    Question: {question}

    Context: {context}

    Answer:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Initialize the LLM
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0)
    
    # --- THIS IS THE FIX ---
    # The code now handles a list containing both Document objects and strings.
    if documents:
        processed_docs = []
        for doc in documents:
            # If the item is a Document object, get its page_content
            if isinstance(doc, Document):
                processed_docs.append(doc.page_content)
            # If the item is already a string, use it directly
            elif isinstance(doc, str):
                processed_docs.append(doc)
        
        document_contents = "\n\n---\n\n".join(processed_docs)
    else:
        document_contents = "No documents found."
    
    print(f"--- CONTEXT LENGTH: {len(document_contents)} chars ---")

    # Create the generation chain
    rag_chain = prompt | llm | StrOutputParser()

    # Invoke the chain with the formatted context and question
    generation = rag_chain.invoke({"context": document_contents, "question": question})

    print(f"✅ Generated: {generation}")

    # Return the updated state
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "generation_count": generation_count, # Increment the generation count
    }

