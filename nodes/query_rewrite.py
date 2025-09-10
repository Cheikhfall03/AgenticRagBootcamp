from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from state import GraphState

# --- 1. Define the Pydantic Model for Structured Output ---
# This ensures the LLM's output is predictable.
class RewrittenQuestion(BaseModel):
    """A rewritten question that is optimized for vector database retrieval."""
    rewritten_question: str = Field(
        description="A new, standalone question that is improved for vectorstore retrieval."
    )

# --- 2. Define the Prompt for Query Rewriting ---
rewrite_prompt_template = """You are an expert at rewriting user questions to be more effective for a vector database search.
Look at the original user question and rewrite it to be more clear, specific, and descriptive.
Your goal is to improve the chances of retrieving relevant documents.

Original Question: {question}

Provide your rewritten question in a structured format."""

rewrite_prompt = ChatPromptTemplate.from_template(rewrite_prompt_template)

# --- 3. Initialize the Language Model with Structured Output ---
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
structured_llm_rewriter = llm.with_structured_output(RewrittenQuestion)

# --- 4. Define the Complete Query Rewriting Chain ---
query_rewrite_chain = rewrite_prompt | structured_llm_rewriter

def query_rewrite(state: GraphState):
    """
    Rewrites the user's question to improve retrieval accuracy.
    """
    print("---REWRITE QUERY---")
    
    question = state["question"]
    rewrite_count = state.get("query_rewrite_count", 0) + 1

    # Invoke the chain to get the structured output
    rewrite_result = query_rewrite_chain.invoke({"question": question})

    # --- THIS IS THE FIX ---
    # We now correctly extract the string from the Pydantic object.
    rewritten_question_str = rewrite_result.rewritten_question

    print(f"✅ Original Question: {question}")
    print(f"✅ Rewritten Question: {rewritten_question_str}")
    
    # Return the updated state with the rewritten question string
    return {
        "question": rewritten_question_str,
        "documents": [],  # Clear documents to force a new retrieval
        "query_rewrite_count": rewrite_count,
    }

