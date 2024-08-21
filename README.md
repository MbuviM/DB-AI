# Diabetes AI Chatbot

## Overview

The Diabetes AI Chatbot is a web application designed to provide answers and insights related to diabetes. Utilizing advanced natural language processing and text-to-speech capabilities, the chatbot offers interactive and accessible information for users seeking clarification on diabetes-related topics.

## Features

- **Text-Based Interaction**: Users can ask questions related to diabetes and receive detailed text responses.
- **Voice Responses**: The chatbot converts text responses into speech using Google’s TTS, allowing for an engaging, hands-free experience.
- **Data Storage**: Uses TiDB as a vector store for storing and querying diabetes-related information efficiently.
- **Streamlit Interface**: A user-friendly web interface built with Streamlit for easy interaction and visualization.

## Installation

To run this project locally, follow these steps:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/diabetes-ai-chatbot.git
   cd diabetes-ai-chatbot
   ```

2. **Set Up a Virtual Environment**
    ```pip install virtualenv```

   ```bash
   python -m venv venv 
   source .venv/bin/activate  
   ```
   ```windows
   virtualenv .venv 
   .venv/Scripts/activate
   ```
   

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Secrets**

   Create a `.env` file in the root directory and add the following environment variables:

   ```dotenv
   TIDB_USERNAME=your_tidb_username
   TIDB_PASSWORD=your_tidb_password
   TIDB_HOST=your_tidb_host
   ```

   Alternatively, use Streamlit secrets management if deploying on Streamlit Cloud.

5. **Prepare Data**

   Place your data file (`data.txt`) in the root directory or adjust the path in the `prepare_data` function.

6. **Run the Application**

   ```bash
   streamlit run deploy.py
   ```

## Usage

1. **Open the Web Application**

   After running the Streamlit command, open your browser and navigate to `http://localhost:8501`.

2. **Ask Questions**

   Enter your questions related to diabetes into the input field and press "Submit."

3. **Receive Responses**

   The chatbot will display text responses and play audio responses automatically.

## Deployment

This project is deployed on [Streamlit Cloud](https://streamlit.io/cloud). To deploy it yourself, follow these steps:

1. **Push to GitHub**

   Make sure your project is pushed to a GitHub repository.

2. **Deploy on Streamlit Cloud**

   Follow the [Streamlit Cloud deployment guide](https://docs.streamlit.io/streamlit-cloud) to connect your GitHub repository and deploy the app.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue if you have suggestions or improvements.

## Contact

For questions or feedback, please contact [mwendembuvi10@gmail.com].
