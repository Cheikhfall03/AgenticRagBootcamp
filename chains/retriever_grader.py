# chains/retriever_grader.py (Version Corrigée)

# 1. Imports (JsonOutputParser ajouté)
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser # Ajout de l'import
from langchain_core.prompts import ChatPromptTemplate
import os

# 2. Modèle Pydantic (inchangé)
class GradeDocuments(BaseModel):
    """Score binaire pour la vérification de la pertinence des documents récupérés."""
    binary_score: bool = Field(
        description="Le document est-il pertinent pour la question ? Mettre à True si pertinent, False sinon."
    )

# 3. LLM (sans .with_structured_output)
llm = ChatGroq(
    # Pour de meilleures performances, envisagez "llama3-8b-8192"
    model="openai/gpt-oss-20b", 
    temperature=0.0
)

# 4. Prompt (légèrement amélioré pour plus de clarté)
system = """Vous êtes un évaluateur qui juge la pertinence d'un document récupéré par rapport à une question de l'utilisateur.
    Votre objectif est de filtrer les documents non pertinents. Si le document contient des mots-clés ou un sens sémantique lié à la question, notez-le comme pertinent.
    Répondez uniquement avec un objet JSON valide. Fournissez un score booléen : True si le document est pertinent, False sinon."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Document Récupéré: \n\n {document} \n\n Question de l'utilisateur: {question}"),
    ]
)

# 5. Parser JSON
parser = JsonOutputParser(pydantic_object=GradeDocuments)

# 6. Chaîne Finale (CORRIGÉE avec .bind() et le parser)
retrieval_grader = (
    grade_prompt 
    | llm
    | StrOutputParser
)
