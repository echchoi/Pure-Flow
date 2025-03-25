from pinecone import Pinecone
import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Define the index name
sparse_index_name = "sparse-index"
sparse_index_host = os.environ.get("PINECONE_SPARSE_INDEX_HOST")

# Check if index exist and connect to the index
if pc.has_index(index_name):
    sparse_index = pc.Index(host=index_host)

# Example usage
if __name__ == "__main__":
    sparse_index.delete(ids=["1", "2", "3"], namespace='water')
