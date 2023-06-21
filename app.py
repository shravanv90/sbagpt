import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Sidebar contents
with st.sidebar:
    st.title('Chat with Sinequa SBA')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    st.write('[Sinequa SBA Documentation](https://sinequa.github.io/sba-angular/)')
 

def main():
    load_dotenv()
    st.header("Ask about Sinequa SBA")

    # Input field for OpenAI API key
    openai_api_key = st.text_input("Enter your OpenAI API key:")

    if openai_api_key:
        st.write("API key entered.")

    # Check if the sinequa_sba_documentation.pkl file exists
    if os.path.isfile("sinequa_sba_documentation.pkl"):
        # The file exists, so the upload PDF section should not be visible
        st.write("PDF upload section is not available.")
        VectorStore = None
        with open("sinequa_sba_documentation.pkl", "rb") as f:
            VectorStore = pickle.load(f)
        st.write('Embeddings Loaded from the Disk')
        
        if VectorStore is not None:
            # Accept user questions/query
            query = st.text_input("Ask questions about Sinequa SBA:")
            if query:
                docs = VectorStore.similarity_search(query=query, k=3)
 
                llm = OpenAI(temperature=0.7, max_tokens=100, verbose="true", openai_api_key=openai_api_key)
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    print(cb)
                st.markdown('---')
                st.markdown(f'**Response:**')
                st.text_area('', value=response, height=200)

    else:
        # The file doesn't exist, so the upload PDF section is visible
        # upload a PDF file
        pdf = st.file_uploader("Upload your PDF", type='pdf')
 
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
 
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)
 
            # embeddings
            store_name = pdf.name[:-4]
            st.write(f'{store_name}')
 
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
                st.write('Embeddings Loaded from the Disk')
            else:
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

if __name__ == '__main__':
    main()
