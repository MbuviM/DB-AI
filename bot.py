# Import Libraries
import os
import click
from sqlalchemy import URL
from openai import OpenAI
import openai
import scipy.io.wavfile as wavfile
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.tidbvector import TiDBVectorStore
import time
from contextlib import contextmanager
from dotenv import load_dotenv
from io import BytesIO
import streamlit as st

#load_dotenv()
client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

# Define TiDB connection URL
tidb_connection_url = URL(
    "mysql+pymysql",
    username=st.secrets['TIDB_USERNAME'],
    password=st.secrets['TIDB_PASSWORD'],
    host=st.secrets['TIDB_HOST'],
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

# Context manager for temporary files
@contextmanager
def temporary_file(filename):
    try:
        yield filename
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def speak_response(response_text, voice="echo", format="mp3"):
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=response_text
        )
        
        with BytesIO(response.content) as temp_mp3:
            audio = AudioSegment.from_mp3(temp_mp3)
            wav_buffer = BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            print("Audio generated successfully.")
            return wav_buffer
    
    except openai.OpenAIError as e:
        print(f"Error generating speech: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during speech generation: {e}")
        return None
    
def chat_with_voice():
    prepare_data()
  
    question = input(f"Question: ")
    response = query_engine.query(question)
    print(f"Response: {response}")
    speak_response(str(response))

if __name__ == '__main__':
    chat_with_voice()