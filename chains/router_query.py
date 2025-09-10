# chains/router_query.py (Version Corrigée)

from typing import Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser # Ajout de l'import
from langchain_core.prompts import ChatPromptTemplate
import os

# Modèle Pydantic (inchangé)
class RouteQuery(BaseModel):
    """Route une question de l'utilisateur vers la source de données la plus pertinente."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Étant donné une question, choisir de la router vers 'web_search' ou 'vectorstore'.",
    )

# LLM (sans .with_structured_output)
llm = ChatGroq(
    # Pour de meilleures performances, envisagez "llama3-8b-8192"
    model="openai/gpt-oss-20b", 
    temperature=0.0
)

# Prompt (inchangé, il est bien formulé)
system = """Vous êtes un expert pour router une question utilisateur vers une base de données vectorielle (vectorstore) ou une recherche web (web_search).
La base de données vectorielle contient des documents sur les agents IA, l'ingénierie des prompts et les attaques adverses.
Utilisez la 'vectorstore' pour les questions sur ces sujets. Pour tout le reste, utilisez 'web_search'."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Parser JSON
parser = JsonOutputParser(pydantic_object=RouteQuery)

# Chaîne Finale (CORRIGÉE avec .bind() et le parser)
question_router = (
    route_prompt 
    | llm.bind(response_format={"type": "json_object"}) 
    | parser
)
