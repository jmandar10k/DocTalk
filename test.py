   import streamlit as st
   from langchain_community.llms import Ollama
   import os

   st.title("Ollama Connection Test")
   try:
     ollama_url = os.environ.get("OLLAMA_URL")
     print(f"Retrieved OLLAMA_URL: {ollama_url}")
     if not ollama_url:
         st.error("OLLAMA_URL not set in Streamlit Secrets")
         st.stop()
     print(f"Base_url used for ollama: {ollama_url}")
     llm = Ollama(model="nemotron-mini", base_url=ollama_url)
     st.success(f"Ollama client initialized successfully")
     st.write(f"Ollama client base URL: {llm.base_url}")
     try:
         st.write(llm.invoke("test connection"))
     except Exception as e:
         st.error(f"Failed to invoke Ollama: {e}")
   except Exception as e:
      st.error(f"An error occurred: {e}")
      st.write(f"Error Details: {e}")
