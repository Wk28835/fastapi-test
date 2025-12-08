import os
from typing import List
from dotenv import load_dotenv

# We will keep load_dotenv to get the API key and provider
load_dotenv(override=True)

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")
# --- FIX: HARDCODE THE CORRECT MODEL TO BYPASS THE OVERRIDE ISSUE ---
EMBEDDING_MODEL = "text-embedding-004"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- ADD THIS DEBUG BLOCK ---
#print(f"DEBUG: Loaded EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")
#print(f"DEBUG: Loaded EMBEDDING_MODEL: {EMBEDDING_MODEL}")
#print(f"DEBUG: GEMINI_API_KEY is present: {bool(GEMINI_API_KEY)}")
# -----------------------------

# -----------------------------
# GEMINI EMBEDDINGS
# -----------------------------
if EMBEDDING_PROVIDER == "gemini":
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Returns embeddings for a list of strings using Gemini.
    """
    if EMBEDDING_PROVIDER != "gemini":
        raise NotImplementedError(f"Embedding provider '{EMBEDDING_PROVIDER}' is not implemented.")

    try:
        # FIXED CALL: Use genai.embed_content
        resp = genai.embed_content(
            model=EMBEDDING_MODEL, # Uses the variable that should be "text-embedding-004"
            content=texts,
            task_type="RETRIEVAL_DOCUMENT"
        )
        
        # Return the list of embeddings
        return resp['embedding']

    except Exception as e:
        # You see this print in the traceback, meaning the API call failed
        print(f"An error occurred during Gemini embedding: {e}") 
        raise # The traceback shows the error is being re-raised successfully