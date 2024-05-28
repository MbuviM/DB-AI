import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Load fine-tuned model and tokenizer for inference
model = AutoModelForSeq2SeqLM.from_pretrained('./fine_tuned_model')
tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')

# Define a chat function for the chatbot
def chat_with_bot(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model.generate(inputs.input_ids, max_length=128, num_beams=4, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI
st.title("Chatbot Interface")
st.write("Interact with the fine-tuned chatbot!")

user_input = st.text_input("You:", "")
if st.button("Send"):
    if user_input:
        response = chat_with_bot(user_input, model, tokenizer)
        st.text_area("Bot:", value=response, height=200)
    else:
        st.write("Please enter a message.")
