import streamlit as st
from bot import prepare_data, query_engine, speak_response
import io

def main():
    # Prepare data before starting the chat
    prepare_data()
    st.set_page_config(
        page_title="Diabetes AI Chatbot",
        page_icon="bot.jpeg",
        layout="centered",
    )
    st.title("Diabetes AI Chatbot")
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
            audio_buffer = speak_response(str(response))

            # Play the audio in Streamlit automatically
            if audio_buffer:
                st.audio(audio_buffer.getvalue(), format='audio/mp3', unsafe_allow_html=True, start_time=0, volume=1.0)

        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()