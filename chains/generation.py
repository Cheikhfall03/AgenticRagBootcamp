import os
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Initialize the language model
llm = ChatGroq(
    temperature=0, 
    model_name="llama3-8b-8192", # Or your preferred model
    api_key=os.environ.get("GROQ_API_KEY")
)

# Define the generation prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant. Use the following context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nContext: {context}"),
    ("human", "Question: {question}"),
])

# Create the generation chain
generation_chain = prompt | llm | StrOutputParser()

def generate(state: dict) -> dict:
    """
    Generates an answer using the retrieved documents and the user's question.
    It truncates the context to a safe limit to prevent API errors.
    """
    print("---NODE: GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # THIS IS THE FIX 👇
    # Join documents and truncate to prevent exceeding the model's context limit.
    # Groq's limit is 6000 TPM. A safe character limit (e.g., 18000 chars)
    # is a good way to stay well under the token limit.
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc in documents])
    
    SAFE_CHARACTER_LIMIT = 18000  # Approx. 4500-5000 tokens
    if len(context_text) > SAFE_CHARACTER_LIMIT:
        print(f"⚠️  Context length ({len(context_text)}) exceeds safe limit. Truncating.")
        context_text = context_text[:SAFE_CHARACTER_LIMIT]

    # Invoke the chain with the potentially truncated context
    try:
        generation = generation_chain.invoke({"context": context_text, "question": question})
        return {"generation": generation}
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return {"generation": "I'm sorry, I encountered an error while generating a response."}
