# Import Libraries
import os
import click
from sqlalchemy import URL
from openai import OpenAI
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.tidbvector import TiDBVectorStore

# Define TiDB connection URL
tidb_connection_url = URL(
    "mysql+pymysql",
    username=os.environ['TIDB_USERNAME'],
    password=os.environ['TIDB_PASSWORD'],
    host=os.environ['TIDB_HOST'],
    port=4000,
    database="DATA",
    query={"ssl_verify_cert": True, "ssl_verify_identity": True},
)

# Initialize TiDB Vector Store
tidbvec = TiDBVectorStore(
    connection_string=tidb_connection_url,
    table_name="diabetes_rag_app",
    distance_strategy="cosine",
    vector_dimension=1536, # The dimension is decided by the model
    drop_existing_table=False,
)

# Create VectorStoreIndex and StorageContext for storage
"""
StorageContext is used to store vectors, nodes and indices while 
VectorStoreIndex is used to accept nodes which are chunks of 
documents and creates indexes for them.
"""
tidb_vec_index = VectorStoreIndex.from_vector_store(tidbvec)
storage_context = StorageContext.from_defaults(vector_store=tidbvec)
query_engine = tidb_vec_index.as_query_engine(streaming=True)

# Function to prepare data
def prepare_data():
    documents_path = "data.txt"
    
    try:
        with open(documents_path, "r", encoding="utf-8") as file:
            documents_text = file.read()
        
        # Convert the raw text into a list of Document objects
        documents = [Document(text=documents_text)]
        
        # Add documents to the vector index
        tidb_vec_index.from_documents(documents, storage_context=storage_context, show_progress=True)
        click.echo("Data preparation complete.")
        
    except FileNotFoundError:
        click.echo(f"Error: The file '{documents_path}' was not found.")
    except Exception as e:
        click.echo(f"An error occurred during data preparation: {e}")

def chat():
    prepare_data()
    question = input("Enter your question: ")
    response = query_engine.query(question)
    click.echo(response)

if __name__ == '__main__':
    chat()