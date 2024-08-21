import streamlit as st
from bot import prepare_data, query_engine, speak_response
from io import BytesIO
import base64

def get_audio_html(audio_buffer):
    """Generate HTML for playing audio with autoplay"""
    audio_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")
    audio_html = f"""
    <audio controls autoplay>
        <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    """
    return audio_html

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
            wav_buffer = speak_response(str(response))

            # Play the audio in Streamlit automatically
            if wav_buffer:
                audio_html = get_audio_html(wav_buffer)
                st.components.v1.html(audio_html, height=0)  # Set height to 0 to hide the player

        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()


