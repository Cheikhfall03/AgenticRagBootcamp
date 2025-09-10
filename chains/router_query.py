# chains/router_query.py (Version Corrigée)

from typing import Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser # Ajout de l'import
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

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

structured_llm_rewriter = llm.with_structured_output(RouteQuery)


# Chaîne Finale (CORRIGÉE avec .bind() et le parser)
question_router = (
    route_prompt 
    | structured_llm_rewriter
)
