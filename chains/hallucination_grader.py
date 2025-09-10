# hallucination_grader.py

from pydantic import BaseModel, Field
from langchain import hub
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os

# ---------------------------
# 1. Définition du schéma Pydantic
# ---------------------------
class GradeHallucinations(BaseModel):
    """Binaire : évalue si la génération est fidèle aux documents"""
    binary_score: bool = Field(
        description="True si la génération est soutenue par les faits, False si c'est une hallucination"
    )

# ---------------------------
# 2. Chargement du LLM
# ---------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# ---------------------------
# 3. Prompt clair avec contrainte JSON
# ---------------------------
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.

- If the generation is fully supported by the retrieved facts, respond with: {"binary_score": true}
- If the generation contains hallucinations or is not supported, respond with: {"binary_score": false}
Only output valid JSON, nothing else.
"""

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("user", "Retrieved facts:\n\n{documents}\n\nGeneration:\n\n{generation}")
])

# ---------------------------
# 4. Parser JSON fiable
# ---------------------------
parser = JsonOutputParser(pydantic_object=GradeHallucinations)

# ---------------------------
# 5. Chaîne finale (prompt -> llm -> parser)
# ---------------------------
hallucination_grader = hallucination_prompt | llm | parser
