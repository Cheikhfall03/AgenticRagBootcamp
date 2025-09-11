# chains/generation.py
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnablePassthrough
# LLM Groq
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # or another supported model
    temperature=0.0
)

# ✅ Strict anti-hallucination prompt
prompt = ChatPromptTemplate.from_template("""
You are a question-answering assistant.
Only use the information provided in the CONTEXT below.
If the answer is not present in the context, strictly respond: "I don't know".

---
CONTEXT:
{documents}
---
QUESTION:
{question}
---
ANSWER (based only on the context):
""")

# Generation pipeline
generation_chain = (
    {"documents": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
