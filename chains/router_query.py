### Router

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from  pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import os

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


# LLM with function call

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY")
)
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
