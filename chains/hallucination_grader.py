# chains/hallucination_grader.py (Version Corrigée)

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# ---------------------------
# 1. Schéma Pydantic (inchangé)
# ---------------------------
class GradeHallucinations(BaseModel):
    """Binaire : évalue si la génération est fidèle aux documents"""
    binary_score: bool = Field(
        description="True si la génération est soutenue par les faits, False si c'est une hallucination"
    )

# ---------------------------
# 2. LLM (inchangé)
# ---------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b", # Note: Ce modèle peut être remplacé par un plus récent comme "llama3-8b-8192" pour de meilleures performances
    temperature=0
)

# ---------------------------
# 3. Prompt (inchangé)
# ---------------------------
# Correction à appliquer dans chains/hallucination_grader.py

# ... autres imports ...
from langchain_core.prompts import ChatPromptTemplate

# ...

# ---------------------------
# 3. Prompt (CORRIGÉ)
# ---------------------------
# Assurez-vous que votre variable system ressemble exactement à ceci :
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.

- If the generation is fully supported by the retrieved facts, respond with: {"binary_score": true}
- If the generation contains hallucinations or is not supported, respond with: {"binary_score": false}
Only output a valid JSON object, nothing else.
"""

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("user", "Retrieved facts:\n\n{documents}\n\nGeneration:\n\n{generation}")
])

# ... le reste du fichier ...

# ---------------------------
# 4. Parser (inchangé)
# ---------------------------
parser = JsonOutputParser(pydantic_object=GradeHallucinations)

# ---------------------------
# 5. Chaîne finale (CORRIGÉE)
# ---------------------------
# On force le LLM à répondre en format JSON en utilisant .bind()
# C'est la correction clé pour éviter l'erreur 'tool_use_failed'.
hallucination_grader = (
    hallucination_prompt 
    | llm.bind(response_format={"type": "json_object"}) 
    | parser
)
