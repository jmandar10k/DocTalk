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

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="DocTalk - Business Document Assistant",
    page_icon="📚",
    layout="wide"
)

# Simple CSS for clean styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #10A37F;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0E8F6D;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        padding: 0.5rem;
    }
    .stMarkdown {
        font-family: 'Helvetica Neue', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 2rem;
    }
    .answer-box {
        padding: 1.5rem;
        border-radius: 5px;
        margin-top: 1rem;
        border: 1px solid #10A37F;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.title("📚 DocTalk")
st.markdown("### Business Document Intelligence Assistant")

# Sidebar
with st.sidebar:
    st.header("Document Upload")
    
    # Features
    st.subheader("Features")
    st.markdown("""
        - 📄 Business Document Analysis
        - 💡 Text Extraction & Summarization
        - 🔍 Document Q&A
    """)
    
    # Guidelines
    st.subheader("Guidelines")
    st.markdown("""
        - Upload PDF documents
        - Text-heavy documents preferred
        - Business reports, contracts, etc.
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload your business document"
    )

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
    # Save uploaded file
    with open("uploaded_document.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("✅ File uploaded successfully!")
    
    # Data Ingestion
    with st.spinner("Processing your document..."):
        loader = PyPDFLoader("uploaded_document.pdf")
        text_documents = loader.load()
        
        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(text_documents)

        # Embedding Setup
        API_KEY = 'AIzaSyA5Dv8GBGaqg_7MhmIIyxHj4qRkAG8s82E'

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

    # Query Section
    st.markdown("---")
    st.subheader("Ask Questions")
    
    query = st.text_input(
        "Enter your question:",
        placeholder="Type your question here...",
        help="Ask any question about your document"
    )
    
    if st.button("Get Answer"):
        if query.strip():
            with st.spinner("Thinking..🤔"):
                response = retrieval_chain.invoke({"input": query})
                st.markdown("""
                    <div class='answer-box'>
                        <h4>Answer:</h4>
                        <p>{}</p>
                    </div>
                """.format(response["answer"]), unsafe_allow_html=True)
        else:
            st.warning("Please enter a valid query.")
else:
    # Welcome message
    st.info("👈 Please upload a PDF document from the sidebar to get started.")

