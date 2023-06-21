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
    
    store_name = "sinequa_sba_documentation"
    
    if not os.path.exists(f"{store_name}.pkl"):
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
            
            embeddings = OpenAIEmbeddings(openai_api_key="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")
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
        else:
            return # Exit if no PDF uploaded
    
    else:
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
        st.write('Embeddings Loaded from the Disk')
 
    openai_key = st.text_input("Enter your OpenAI key:")
    
    if openai_key:
        query = st.text_input("Ask questions about Sinequa SBA:")
        if query:
            docs = VectorStore.similarity_search(query=query, k=3, distances="True", labels="True")
            llm = OpenAI(temperature=0.5, max_tokens = 1000, verbose="true", openai_api_key=openai_key)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.markdown(f'<div style="background-color: #f0f0f0; padding:10px; border-radius: 10px;">{response}</div>', unsafe_allow_html=True)
 
if __name__ == '__main__':
    main()
