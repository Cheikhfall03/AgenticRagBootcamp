# chains/generation.py
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()
