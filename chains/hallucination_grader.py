# ### Hallucination Grader ###

# 1. Imports
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
# 2. Data model (Corrected to use a boolean for a reliable binary score)
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in the generation."""

    binary_score: bool = Field(
        description="Is the answer grounded in the facts? Set to True if grounded, False otherwise."
    )

# 3. LLM with structured output
# Using a specific model for grading is a good practice.
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.0,
   api_key=os.getenv("GROQ_API_KEY")
)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# 4. Prompt (Corrected to ask for a boolean score)
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
     Provide a boolean score. Set the score to True if the answer is grounded in the set of facts, and False otherwise."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

# 5. Final chain
hallucination_grader = hallucination_prompt | structured_llm_grader

