"""
Process Markdown FAQ file by splitting it into sections based on headers.

This script reads a Markdown FAQ file and splits it into sections using headers as delimiters.
The output shows each section with its corresponding header.
"""

from langchain_text_splitters import MarkdownHeaderTextSplitter
from pinecone import Pinecone, Index
import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()

# Setup Pinecone
NAMESPACE = "water"
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
sparse_index_name = os.environ.get("PINECONE_SPARSE_INDEX_NAME")
dense_index_name = os.environ.get("PINECONE_DENSE_INDEX_NAME")
spase_index_host = os.environ.get("PINECONE_SPARSE_INDEX_HOST")
dense_index_host = os.environ.get("PINECONE_DENSE_INDEX_HOST")

def check_and_create_index(index_name: str, dimension: int, metric: str, spec: dict):
    """
    Check if the index exists and prompt error message and stop the script if it doesn't.
    
    Args:
        index_name (str): The name of the index to check or create.
        dimension (int): The dimension of the vectors.
        metric (str): The metric to use for similarity search.
        spec (dict): The specification for the index.
    """
    try:
        pc.describe_index(index_name)
    except Exception:
        print("Exception: {index_name} does not exist")
        return False

def split_markdown(md_string):
    """
    Split Markdown FAQ text into sections based on headers.
    
    Args:
        md_string (str): The Markdown FAQ text to be split
        
    Returns:
        list: A list of Document objects containing the split sections
    """
    # Define headers to split on
    headers_to_split_on = [
        ("#", "Header 1"),  # Main headers
        ("##", "Header 2"),  # Subheaders
    ]

    # Initialize the Markdown header text splitter
    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # Keep headers in the output
    )

    # Split the text into sections
    splits = text_splitter.split_text(md_string)
    return splits

def upsert_record(index_name: str, records: list):
    """
    Upsert records into the specified index.
    
    Args:
        index_name (str): The name of the index to upsert into
        records (list): A list of records to be upserted
    """
    # Convert Document objects to Pinecone records
    pinecone_records = []
    for i, doc in enumerate(records):
        pinecone_records.append({
            "id": f"doc_{i}",
            "chunk_text": doc.page_content,
            "source": doc.metadata.get("source", f"doc_{i}")
        })
    
    if pc.has_index(index_name):
        index = pc.Index(index_name)
        index.upsert_records(
            namespace=NAMESPACE,
            records=pinecone_records
        )

def main(markdown_file_path):
    """
    Main function to process the Markdown file.
    """
    # Read the Markdown file
    with open(markdown_file_path, 'r') as file:
        md_string = file.read()

    # Split the Markdown into sections
    md_splits = split_markdown(md_string)

    # Upsert records into the index
    upsert_record(dense_index_name, md_splits)
    upsert_record(sparse_index_name, md_splits)

if __name__ == "__main__":
    main('Documents/FAQ.md')
