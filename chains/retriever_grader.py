# chains/retriever_grader.py (Non-JSON Version)

# 1. Imports
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
import os

# 2. Pydantic Model (This is still needed for the graph's state)
class GradeDocuments(BaseModel):
    """Binary score for document relevance check."""
    binary_score: bool = Field(
        description="Is the document relevant to the question? True if relevant, False otherwise."
    )

# 3. LLM
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.0
)

# 4. New prompt asking for 'yes' or 'no'
system = """You are a strict relevance grader. Your goal is to determine if the document is useful for answering the user's question.

Respond with only a single word: 'yes' or 'no'.
- 'yes' if the document provides a direct answer or crucial context.
- 'no' if it is off-topic or a passing mention."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
    ]
)

# --- NEW CHAIN LOGIC (NON-JSON) ---

# 5. Function to convert string "yes" or "no" to a boolean
def text_to_boolean(text: str) -> bool:
    return "yes" in text.lower()

# 6. Final Chain
# This chain now does the following:
#   - Gets a string ("yes" or "no") from the LLM.
#   - Converts that string to a boolean (True/False).
#   - Wraps the boolean in the GradeDocuments model to match the graph's state.
retrieval_grader = (
    grade_prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(text_to_boolean)
    | RunnableLambda(lambda is_relevant: GradeDocuments(binary_score=is_relevant))
)
