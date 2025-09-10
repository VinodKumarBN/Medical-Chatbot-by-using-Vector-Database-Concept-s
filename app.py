# streamlit_app.py
import streamlit as st
from src.helper import ChatCohere, local_emb
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

# Load API keys
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medical-chatbot"

# Vector store
vec_store = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=local_emb
)

retriever = vec_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Cohere LLM
cohere_chat = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-a-03-2025"
)

# Retrieval QA with custom prompt
qa = RetrievalQA.from_chain_type(
    llm=cohere_chat,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": qa_prompt}
)

# ---- Streamlit UI ----
st.set_page_config(page_title="💊 Medical Chatbot", page_icon="💊")

st.title("💊 Medical Chatbot")
st.write("Ask me anything about medical topics.")

user_input = st.text_input("Your question:")

if st.button("Ask") and user_input:
    with st.spinner("Thinking..."):
        result = qa({"query": user_input})
    st.success(result["result"])



