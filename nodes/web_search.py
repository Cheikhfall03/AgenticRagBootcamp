import os
import time
from dotenv import load_dotenv
# --- THIS IS THE FIX ---
# Updated import to use the non-deprecated TavilySearch from the correct package
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from state import GraphState

# Load environment variables
load_dotenv()

# Validate Tavily API Key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("❌ TAVILY_API_KEY environment variable is not set! Add it to your .env file.")

# --- THIS IS THE FIX ---
# Initialize the modern, non-deprecated TavilySearch tool
tavily_tool = TavilySearch(max_results=5)

def web_search(state: GraphState):
    """
    Performs a web search using Tavily API and handles different output formats.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    try:
        # Small delay to avoid rate limits
        time.sleep(0.5)

        # Invoke the search tool
        search_output = tavily_tool.invoke(question)

        # The code now correctly handles the dictionary output from Tavily.
        results_list = []
        if isinstance(search_output, dict) and 'results' in search_output:
            # If the output is a dictionary, extract the list from the 'results' key
            results_list = search_output['results']
            print(f"🔍 Extracted {len(results_list)} results from the Tavily dictionary.")
        elif isinstance(search_output, list):
            # Also handle the case where it might return a list directly
            results_list = search_output
            print("🔍 Received a direct list from Tavily.")

        # Process the extracted list of results
        web_docs = []
        for result in results_list:
            if isinstance(result, dict):
                # Convert each result dictionary into a LangChain Document object
                web_docs.append(Document(
                    page_content=result.get("content", ""),
                    metadata={
                        "source": result.get("url", "N/A"),
                        "score": result.get("score", "N/A")
                    }
                ))

        print(f"✅ Created {len(web_docs)} Document objects from web search.")
        
        # Append the new Document objects to the state
        all_documents = documents + web_docs

        return {
            "documents": all_documents,
            "question": question,
        }

    except Exception as e:
        print(f"❌ ERROR in Tavily search: {e}")
        # Return the original state safely
        return {
            "documents": documents,
            "question": question,
        }

