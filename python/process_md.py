"""
Process Markdown FAQ file by splitting it into sections based on headers.

This script reads a Markdown FAQ file and splits it into sections using headers as delimiters.
The output shows each section with its corresponding header.
"""

from langchain_text_splitters import MarkdownHeaderTextSplitter
import re
from pinecone import Pinecone, Index
import os

# Setup Pinecone
NAMESPACE = "default"
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

def check_and_create_index(index_name: str, dimension: int, metric: str, spec: dict):
    """
    Check if the index exists and create it if it doesn't.
    
    Args:
        index_name (str): The name of the index to check or create.
        dimension (int): The dimension of the vectors.
        metric (str): The metric to use for similarity search.
        spec (dict): The specification for the index.
    """
    try:
        pc.describe_index(index_name)
    except Exception:
        pc.create_index(name=index_name, dimension=dimension, metric=metric, spec=spec)

def split_markdown(faq_string):
    """
    Split Markdown FAQ text into sections based on headers.
    
    Args:
        faq_string (str): The Markdown FAQ text to be split
        
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
    faq_splits = text_splitter.split_text(faq_string)
    return faq_splits

def upsert_record(records: list, index_name: str):
    """
    Upsert records into the specified index.
    
    Args:
        records (list): A list of records to be upserted
        index_name (str): The name of the index to upsert into
    """
    # Convert Document objects to Pinecone records
    pinecone_records = []
    for i, doc in enumerate(records):
        pinecone_records.append({
            "id": f"doc_{i}",
            "values": [0.0] * 1536,  # Placeholder for dense vector
            "metadata": doc.metadata,
            "sparse_values": {"indices": [0], "values": [0.0]}  # Placeholder for sparse vector
        })
    
    index = pc.Index(index_name)
    index.upsert(vectors=pinecone_records, namespace=NAMESPACE)

def main(markdown_file_path):
    """
    Main function to process the Markdown file.
    """
    # Read the Markdown file
    with open(markdown_file_path, 'r') as file:
        md_string = file.read()

    # Split the Markdown into sections
    md_splits = split_markdown(md_string)

    # Check and create index if it doesn't exist
    index_name = "faq-index"
    dimension = 1536
    metric = "cosine"
    spec = {"pod_type": "p1", "replicas": 1, "shards": 1}
    check_and_create_index(index_name, dimension, metric, spec)

    # Upsert records into the index
    upsert_record(md_splits, index_name)

if __name__ == "__main__":
    main('../Documents/FAQ.md')
