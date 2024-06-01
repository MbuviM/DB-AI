import streamlit as st
from transformers import pipeline

# Load fine-tuned model
model = pipeline("text-generation", model="fine_tuned_model")

# Streamlit UI
st.title("Conversational AI Demo")

# Input prompt
prompt = st.text_input("Enter your prompt:")

# Generate response
if st.button("Generate Response"):
    response = model(prompt)[0]["generated_text"]
    st.text_area("Model Response:", value=response, height=200)
