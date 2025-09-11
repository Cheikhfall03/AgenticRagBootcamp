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
