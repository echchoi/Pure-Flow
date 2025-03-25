from pinecone import (
    Pinecone,
    ServerlessSpec
)
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Setup Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
sparse_index_name = os.environ.get("PINECONE_SPARSE_INDEX_NAME")
dense_index_name = os.environ.get("PINECONE_DENSE_INDEX_NAME")

# Check if sparse "):indexes exist
if pc.has_index(sparse_index_name):
    pc.delete_index(sparse_index_name)  # Delete index if it exists

# Create sparse index
pc.create_index_for_model(
    name=sparse_index_name,
    cloud="aws",
    region="us-east-1",
    embed={
        "model":"pinecone-sparse-english-v0",
        "field_map":{"text": "chunk_text"}
    }
)
sparse_index = pc.Index(sparse_index_name)
print(sparse_index.describe_index_stats())

# Check if sparse indexes exist
if pc.has_index(dense_index_name):
    pc.delete_index(dense_index_name)  # Delete index if it exists

# Create dense index
pc.create_index_for_model(
    name=dense_index_name,
    cloud="aws",
    region="us-east-1",
    embed={
        "model":"multilingual-e5-large",
        "field_map":{"text": "chunk_text"}
    }
)
dense_index = pc.Index(dense_index_name)  # Get index handle
print(dense_index.describe_index_stats())
