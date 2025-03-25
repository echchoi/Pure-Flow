from pinecone import Pinecone, ServerlessSpec
import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Define the index name
index_name = "sparse-index"
index_host = os.environ.get("PINECONE_SPARSE_INDEX_HOST")

# Check if index exist and connect to the index
if pc.has_index(index_name):
    sparse_index = pc.Index(host=index_host)

def upsert_text_chunks(chunks):
    """
    Upserts text chunks into the Pinecone index.

    Args:
        chunks (list of dict): List of dictionaries containing 'id' and 'text' keys.
    """
    print(sparse_index.describe_index_stats())
    sparse_index.upsert_records(
        namespace="water",
        records=chunks
    )
    print("Upserted text chunks into the index.")
    print(sparse_index.describe_index_stats())

# Example usage
if __name__ == "__main__":
    text_chunks = [
        {"id": "1", "chunk_text": "This is the first text chunk.", "source": "test_upsert.py"},
        {"id": "2", "chunk_text": "This is the second text chunk.", "source": "test_upsert.py"},
        {"id": "3", "chunk_text": "This is the third text chunk.", "source": "test_upsert.py"},
    ]
    upsert_text_chunks(text_chunks)
