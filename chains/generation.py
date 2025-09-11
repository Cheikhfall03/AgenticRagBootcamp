import os
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Initialize the language model
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant", # Or your preferred model
    api_key=os.environ.get("GROQ_API_KEY")
)

# Define the generation prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant. Use the following context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nContext: {context}"),
    ("human", "Question: {question}"),
])

# Create the generation chain
generation_chain = prompt | llm | StrOutputParser()

