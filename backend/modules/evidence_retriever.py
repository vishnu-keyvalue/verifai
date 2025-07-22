import os
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

def search_for_evidence(query: str, **kwargs):
    """
    Searches the web for evidence related to a query using the Google Custom Search API.

    Args:
        query (str): The claim or text to search for.

    Returns:
        A list of search results or an error dictionary.
    """
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return {"error": "Google API credentials are not configured in the .env file."}
    
    try:
        # Build the custom search service
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        
        # Perform the search
        # We add "fact check" to the query to prioritize results from fact-checking sites
        search_query = f'"{query}" fact check'

        result = service.cse().list(q=search_query, cx=GOOGLE_CSE_ID, **kwargs).execute()

        # Extract the relevant parts of the search results
        search_items = result.get("items", [])
        
        evidence = []
        for item in search_items:
            evidence.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            })

        return {"evidence": evidence}

    except Exception as e:
        print(f"An error occurred during Google search: {e}")
        return {"error": "Failed to retrieve evidence from search engine."}