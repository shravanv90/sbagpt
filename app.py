import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import json

# Sidebar contents
with st.sidebar:
    st.title('Chat with Sinequa SBA')
    st.markdown('''
    ## About
    This app is an AI-driven assistant, capable of answering your questions about Sinequa SBA.
    It utilizes the latest language models from OpenAI and presents a user-friendly interface thanks to Streamlit.
    Feel free to start a conversation with it!
    ''')
    st.write('[Sinequa SBA Documentation](https://sinequa.github.io/sba-angular/)')

# Directory to save chat sessions
CHAT_DIR = "chats"

def save_chat(chat_id, chat):
    """Saves a chat session to a JSON file."""
    os.makedirs(CHAT_DIR, exist_ok=True)
    with open(os.path.join(CHAT_DIR, f"{chat_id}.json"), "w") as f:
        json.dump(chat, f)

def load_chat(chat_id):
    """Loads a chat session from a JSON file."""
    with open(os.path.join(CHAT_DIR, f"{chat_id}.json"), "r") as f:
        return json.load(f)

def list_chats():
    """Lists the IDs of all saved chat sessions."""
    return [filename[:-5] for filename in os.listdir(CHAT_DIR) if filename.endswith(".json")]

def main():
    load_dotenv()
    st.header("Sinequa SBA AI Assistant")

    # Input field for OpenAI API key
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API key:")

    # Check if the API key was entered
    if openai_api_key:
        VectorStore = None
        try:
            with open("sinequa_sba_documentation.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        except Exception as e:
            st.error(f"An error occurred while initializing the assistant. Please make sure the documentation embeddings are available.")
            return

        if VectorStore is not None:
            # Select or start a chat session
            chat_id = st.sidebar.selectbox("Select a chat session or start a new one", list_chats() + ["New"])
            if chat_id == "New":
                chat_id = str(len(list_chats()) + 1)
                chat = []
            else:
                chat = load_chat(chat_id)
            
            chat_placeholder = st.empty()

            # Accept user questions/query
            query = st.text_input("How can I assist you today?")
            if query:
                # Add the user's question to the chat
                chat.append({
                    "sender": "user",
                    "text": query
                })

                try:
                    docs = VectorStore.similarity_search(query=query, k=3)
                    llm = OpenAI(temperature=0.7, max_tokens=100, verbose="true", openai_api_key=openai_api_key)
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=query)
                    
                    # Add the bot's response to the chat
                    chat.append({
                        "sender": "bot",
                        "text": response
                    })

                    if response is not None:
                        if "```typescript" in response or "```cs" in response or "```html" in response:
                            # Add syntax highlighting to the code.
                            formatted_response = f"```typescript\n{response}\n```"
                            chat_placeholder.markdown(formatted_response, unsafe_allow_html=True)
                        else:
                            # Generate a light grey box using HTML and CSS.
                            box_style = """
                            <style>
                            .box {
                                border: solid 1px #ddd;
                                background-color: #f0f0f0;
                                padding: 20px;
                                margin: 20px 0;
                                border-radius: 5px;
                            }
                            </style>
                            """
                            chat_placeholder.markdown(f"{box_style}<div class='box'>{response}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred while processing your question. Please try again.")
                    return

                # Display the chat history
                for message in chat:
                    if message["sender"] == "user":
                        chat_placeholder.markdown(f"<div style='text-align: right; color: blue;'>You: {message['text']}</div>", unsafe_allow_html=True)
                    else:
                        chat_placeholder.markdown(f"<div style='text-align: left; color: green;'>Sinequa SBA: {message['text']}</div>", unsafe_allow_html=True)
                
                # Save the chat
                save_chat(chat_id, chat)

if __name__ == '__main__':
    main()
