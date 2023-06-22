# Sinequa AI Assistant

Sinequa AI Assistant is a chatbot application capable of answering your questions about Sinequa SBA. It uses the latest language models from OpenAI and presents a user-friendly interface thanks to Streamlit.

![Sinequa SBA AI Assistant](screenshot.png)

## Getting Started

Follow these steps to run Sinequa SBA AI Assistant on your local machine.

### Prerequisites

- Python 3.7+
- [Streamlit](https://streamlit.io/)
- [OpenAI](https://www.openai.com/)
- Other Python dependencies from `requirements.txt`

### Installation

1. Clone the repository:
-git clone https://github.com/your-username/sinequa-sba-ai-assistant.git
-cd sinequa-sba-ai-assistant
 
2. Install Python dependencies:
-pip install -r requirements.txt


3. Get your OpenAI API key from the [OpenAI website](https://www.openai.com/). You'll need to create an account and follow their instructions to generate an API key.

4. Run the application:
-streamlit run app.py


5. The Streamlit app should now be running on your local machine. You'll see an URL in your terminal (usually `http://localhost:8501`), open that URL in your browser to use the application.

6. Enter your OpenAI API key in the sidebar of the application and start chatting with the assistant!

## Usage

Enter your query into the text box and press enter. The AI Assistant will then use OpenAI's language model to answer your question based on the loaded PKL files. If you want to change the PKL files, simply replace the ones in the `data` folder with your own. These are generated using vector embedding models from OpenAI.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.






