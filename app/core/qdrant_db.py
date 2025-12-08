import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "panaversity_book_v1")

# Gemini "text-embedding-004" → 768 dimensions
VECTOR_SIZE = 768  

# Initialize Qdrant client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


def ensure_collection():
    """Ensure the Qdrant collection exists with correct vector size."""

    collections = client.get_collections().collections
    existing_names = [c.name for c in collections]

    # If collection does not exist → create
    if COLLECTION not in existing_names:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"[QDRANT] Created collection: {COLLECTION}")

    else:
        print(f"[QDRANT] Collection already exists: {COLLECTION}")
