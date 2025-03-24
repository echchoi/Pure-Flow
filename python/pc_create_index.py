from pinecone import (
    Pinecone,
    ServerlessSpec,
)
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(".env")

# Setup Pinecone
NAMESPACE = "default"
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Create index
if "sparse-index" not in pc.list_indexes():
    pc.create_index(
        name="sparse-index",
        dimension=512,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    sparse_index = pc.Index("sparse-index")

# Create dense index
if "dense-index" not in pc.list_indexes():
    pc.create_index(
        name="dense-index",
        dimension=1024,  # llama-text-embed-v2 output size
        metric="cosine",  # Common for dense vectors
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    dense_index = pc.Index("dense-index")  # Get index handle

print(f"Sparse index created: {sparse_index}")
print(f"Dense index created: {dense_index}")
