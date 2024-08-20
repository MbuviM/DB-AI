# Import Libraries
import os
import click
from sqlalchemy import URL
from openai import OpenAI
import openai
import scipy.io.wavfile as wavfile
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.tidbvector import TiDBVectorStore
import time
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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

# Context manager for temporary files
@contextmanager
def temporary_file(filename):
    try:
        yield filename
    finally:
        if os.path.exists(filename):
            os.remove(filename)

# Convert Text to Speech using OpenAI TTS
def speak_response(response_text, voice="echo", format="mp3"):
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=response_text
        )
        
        # Use a context manager for the temporary MP3 file
        with temporary_file("response.mp3") as temp_mp3:
            with open(temp_mp3, "wb") as audio_file:
                audio_file.write(response.content)
            
            try:
                # Convert MP3 to WAV
                audio = AudioSegment.from_mp3(temp_mp3)
                audio.export("response.wav", format="wav")
                
                # Play the WAV file using sounddevice
                sample_rate, data = wavfile.read("response.wav")
                sd.play(data, sample_rate)
                sd.wait()  # Wait until the audio is finished playing
                
                # Remove the temporary WAV file
                os.remove("response.wav")
            except Exception as e:
                print(f"Error playing audio: {e}")
    
    except openai.OpenAIError as e:
        print(f"Error generating speech: {e}")
    except Exception as e:
        print(f"An error occurred during speech generation: {e}")

def chat_with_voice():
    prepare_data()
  
    question = input(f"Question: ")
    response = query_engine.query(question)
    print(f"Response: {response}")
    speak_response(str(response))

if __name__ == '__main__':
    chat_with_voice()