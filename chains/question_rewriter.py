# ### Question Rewriter ###

# 1. Imports
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import os
# 2. Data model for structured output
class RewriteQuestion(BaseModel):
    """Rewritten question optimized for retrieval."""
    rewritten_question: str = Field(
        ...,
        description="A new, standalone question that is improved for vectorstore retrieval, taking chat history into account.",
    )

# 3. LLM with structured output
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY")
)
structured_llm_rewriter = llm.with_structured_output(RewriteQuestion)

# 4. Prompt (Corrected to include chat history)
system = """You are a question re-writer. Your task is to convert a given question, potentially a follow-up, into a better, standalone question that is optimized for vectorstore retrieval.
Use the provided chat history to understand the context and resolve any ambiguities or references in the latest question."""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "Based on our conversation, rewrite the following question to be a clear, standalone question: \n\n {question}",
        ),
    ]
)

# 5. Parser JSON

# 6. Chaîne Finale (CORRIGÉE avec .bind() et le parser)
question_rewriter = (
    re_write_prompt 
    | structured_llm_rewriter = llm.with_structured_output(RewriteQuestion)
    | StrOutputParser

)

