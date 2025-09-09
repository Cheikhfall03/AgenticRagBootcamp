# ### Retrieval Grader ###

# 1. Imports
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
# 2. Data model (Corrected to use a boolean)
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: bool = Field(
        description="Is the document relevant to the question? Set to True if relevant, False otherwise."
    )

# 3. LLM with structured output

llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY")
)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 4. Prompt (Corrected to ask for a boolean score)
system = """You are a grader assessing the relevance of a retrieved document to a user's question.
    Your goal is to filter out erroneous retrievals. If the document contains keywords or semantic meaning related to the user's question, grade it as relevant.
    Provide a boolean score: True if the document is relevant, and False otherwise."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# 5. Final chain
retrieval_grader = grade_prompt | structured_llm_grader


