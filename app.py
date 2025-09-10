from flask import Flask, render_template, request
from src.helper import ChatCohere, local_emb
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# Load env vars
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "medical-chatbot"

try:
    # Vector store
    vec_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=local_emb
    )

    retriever = vec_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Cohere LLM
    cohere_chat = ChatCohere(
        cohere_api_key=COHERE_API_KEY,
        model="command-a-03-2025"
    )

    # Retrieval QA
    qa = RetrievalQA.from_chain_type(
        llm=cohere_chat,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt}
    )
except Exception as e:
    print("⚠️ Initialization failed:", e)
    qa = None   # so app still runs

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def get_response():
    msg = request.form["msg"]
    if qa is None:
        return "Backend not initialized properly"
    result = qa({"query": msg})
    return str(result["result"])

# ✅ Only runs if executed directly (not on import)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
