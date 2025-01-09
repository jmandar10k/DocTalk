import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate  # Fixed incorrect import for prompt
from langchain_community.llms import Ollama
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import os
import requests  # Added requests for custom timeout handling

st.title("DocTalk- Interactive with Document")
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

try:
    # Get the Ollama URL from environment variable
    ollama_url = os.environ.get("OLLAMA_URL")
    if not ollama_url:
        st.error("OLLAMA_URL environment variable not set. Please configure it in Streamlit Secrets.")
        st.stop()

    if uploaded_file:
        # Save uploaded file temporarily
        with open("uploaded_document.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success("File uploaded successfully!")

        # Data Ingestion
        loader = PyPDFLoader("uploaded_document.pdf")
        text_documents = loader.load()

        # Chunking the document into smaller pieces
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(text_documents)

        # Embedding Setup - Use Streamlit's secrets to access the API key
        API_KEY = st.secrets["GOOGLE_API_KEY"]

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=API_KEY,
        )

        # Creating VectorDB with FAISS
        db = FAISS.from_documents(documents, embeddings)
        retriever = db.as_retriever()

        # LLM Setup - Use the Ollama URL
        llm = Ollama(model="nemotron-mini", base_url=ollama_url)

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
                with st.spinner("Thinking..."):  # Show a spinner while waiting
                    try:
                        # Use a timeout for the Ollama API call
                        response = retrieval_chain.invoke({"input": query}, timeout=30)  # 30 seconds timeout
                        st.subheader("Answer")
                        st.write(response["answer"])
                    except requests.exceptions.Timeout:
                        st.error("The request timed out. Please try again later.")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter a valid query.")
    else:
        st.sidebar.info("Awaiting file upload.")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.write(f"Error Details: {e}")  # Optionally display error details for debugging.
