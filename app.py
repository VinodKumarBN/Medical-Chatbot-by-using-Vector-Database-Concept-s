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

vec_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=local_emb
)

retriever = vec_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

cohere_chat = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-a-03-2025"
)

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
def get_response():
    msg = request.form["msg"]
    print("User:", msg)
    result = qa({"query": msg})
    print("Response:", result["result"])
    return str(result["result"])

import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)




