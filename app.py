from flask import Flask, render_template, jsonify, request
from src.helper import ChatCohere, local_emb
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "medical-chatbot"

# Vector store with HuggingFace embeddings
vec_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=local_emb
)

# Retriever
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

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def get_response():   # ✅ renamed
    msg = request.form["msg"]
    print("User:", msg)
    result = qa({"query": msg})
    print("Response:", result["result"])
    return str(result["result"])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
