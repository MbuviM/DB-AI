import streamlit as st
from bot import prepare_data, query_engine, speak_response
import os

def main():
    # Prepare data before starting the chat
    prepare_data()

    st.set_page_config(
        page_title="Diabetes AI Chatbot",
        page_icon=":syringe:",
        layout="centered",
    )

    st.title("Diabetes AI Chatbot :syringe:")
    st.write("Ask questions related to diabetes and get clarification.")

    # User input area
    user_input = st.text_input("Enter your question below:", placeholder="Type your question here...")

    if st.button("Submit"):
        if user_input:
            # Generate a response using the query engine from bot.py
            with st.spinner("Thinking..."):
                response = query_engine.query(user_input)
            
            # Display the response
            st.markdown(response)

            # Generate audio response
            audio_path = speak_response(str(response))

            # Play the audio in Streamlit
            if os.path.exists(audio_path):
                with open(audio_path, "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/wav")  # Specify the correct format based on your audio

        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()


