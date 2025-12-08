import os
import glob
from embeddings import embed_texts
from qdrant_db import client, COLLECTION, ensure_collection
from bs4 import BeautifulSoup

# Import necessary model structures for clarity and correctness
# This assumes you have PointStruct available, likely from qdrant_client.models
from qdrant_client.models import PointStruct 

# Ensure collection exists
ensure_collection()

# Path to your docs folder
DOCS_PATH = os.getenv("BASE_DOCS_PATH", "../book/docs")

# Collect all .md and .mdx files
files = glob.glob(f"{DOCS_PATH}/**/*.md*", recursive=True)

#print(f"Found {len(files)} files to ingest.")

# -----------------------------------------------------------
# 🌟 CORRECTION: Initialize a global ID counter for Qdrant
# Qdrant requires integer or UUID IDs.
# -----------------------------------------------------------
current_global_id = 0

for filepath in files:
    # Read file content
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove MDX tags and HTML if needed
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text()

    # Split into chunks if large (optional)
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

    # Generate embeddings for each chunk
    embeddings = embed_texts(chunks)

    # Upsert into Qdrant
    points = []
    
    # -----------------------------------------------------------
    # 🌟 CORRECTION: Loop with PointStruct and assign integer ID
    # -----------------------------------------------------------
    for chunk_index, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        # Increment the global ID counter for a unique ID
        current_global_id += 1
        
        point = PointStruct(
            # Use the integer ID
            id=current_global_id, 
            vector=emb,
            payload={
                # Keep the original string ID in the payload for tracing!
                "doc_id_string": f"{os.path.basename(filepath)}_{chunk_index}",
                "doc_path": filepath,
                "text": chunk
            }
        )
        points.append(point)

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Ingested {len(points)} chunks from {filepath} (up to ID: {current_global_id})")

print("✅ All documents ingested into Qdrant successfully!")