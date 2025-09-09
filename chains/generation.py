# chains/generation.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# LLM
llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)

# Prompt — expects {context} and {question}
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Keep the answer concise.\n\n"
            "Context:\n{context}"
        ),
        ("human", "{question}"),
    ]
)

# Simple RAG chain — returns string
rag_chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()  # ← Returns string, not structured model
)