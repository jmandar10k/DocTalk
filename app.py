import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import os
from langchain_groq import ChatGroq

st.title(" DocTalk- Interact with document")
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])


# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ GROQ_API_KEY not found in .env file!")
    st.stop()
# Initialize LLM
def initialize_llm():
    return ChatGroq(
        temperature=0.3,
        model_name="llama-3.1-8b-instant",
        groq_api_key=api_key,
        max_tokens=2000,
        top_p=0.9,
        streaming=True
    )

llm = initialize_llm()



if uploaded_file:
        with open("uploaded_document.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success("File uploaded successfully!")
        
        # Data Ingestion
        loader = PyPDFLoader("uploaded_document.pdf")
        text_documents = loader.load()
        
        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(text_documents)

        # Embedding Setup
        # Use the embedded API key directly
        API_KEY = 'AIzaSyA5Dv8GBGaqg_7MhmIIyxHj4qRkAG8s82E' # Access the API key from Streamlit secrets

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=API_KEY,
        )

        # Creating VectorDB
        db = FAISS.from_documents(documents, embeddings)
        retriever = db.as_retriever()

    

        # Chat Prompt Template
        prompt = ChatPromptTemplate.from_template(""" 
            Answer the following questions based only on provided context.
            Think step by step before providing the detailed answer.
            <context>
            {context}
            </context>
            Question:{input}
        """)

        # Document Chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Retrieval Chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Interactive Query Section
        st.header("Ask Your Question")
        query = st.text_input("Enter your query:")
        
        if st.button("Get Answer"):
            if query.strip():
                with st.spinner("Thinking..."): # Show a spinner while waiting
                    response = retrieval_chain.invoke({"input": query})
                    st.subheader("Answer")
                    st.write(response["answer"])
            else:
                st.warning("Please enter a valid query.")
else:
        st.sidebar.info("Awaiting file upload.")

