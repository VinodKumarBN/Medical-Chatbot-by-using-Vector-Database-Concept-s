# app.py
import os
import logging
import traceback
from threading import Thread
from time import sleep
from flask import Flask, render_template, request, jsonify

# --- imports that may fail if deps/env missing ---
# Import them lazily inside init_services() to avoid import-time failures.

app = Flask(_name_)
logging.basicConfig(level=logging.INFO)

# Global state for initialization
init_state = {
    "ready": False,
    "error": None,
    "started": False
}

# Configuration (env)
INDEX_NAME = os.getenv("PINECONE_INDEX", "medical-chatbot")

def init_services():
    """Run heavy / network initializations here in background."""
    if init_state["started"]:
        return
    init_state["started"] = True
    try:
        logging.info("Init: starting background service initialization...")
        # import here so missing packages don't break module import
        from dotenv import load_dotenv
        load_dotenv()

        COHERE_API_KEY = os.getenv("COHERE_API_KEY")
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        if not COHERE_API_KEY or not PINECONE_API_KEY:
            raise RuntimeError("Missing COHERE_API_KEY or PINECONE_API_KEY in environment")

        # Lazy imports for heavy packages
        from langchain_pinecone import PineconeVectorStore
        from langchain.chains import RetrievalQA
        # your helper provides local_emb or you can import embeddings directly
        from src.helper import local_emb     # adjust if needed
        from src.prompt import qa_prompt
        from langchain_cohere import ChatCohere

        # Initialize Pinecone store (non-blocking, but may raise)
        logging.info("Init: creating/retrieving Pinecone vector store...")
        vec_store = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=local_emb
        )
        retriever = vec_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        logging.info("Init: Pinecone retriever ready.")

        # Initialize Cohere LLM
        logging.info("Init: creating Cohere client...")
        cohere_chat = ChatCohere(cohere_api_key=COHERE_API_KEY, model=os.getenv("COHERE_MODEL", "command-a-03-2025"))
        logging.info("Init: Cohere client ready.")

        # Build QA chain
        logging.info("Init: building RetrievalQA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=cohere_chat,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": qa_prompt}
        )

        # store in global so handlers can use it
        init_state["qa"] = qa_chain
        init_state["ready"] = True
        logging.info("Init: all backend services initialized successfully.")
    except Exception as e:
        tb = traceback.format_exc()
        logging.error("Init: failed:\n" + tb)
        init_state["error"] = tb
        init_state["ready"] = False

# Start background initialization thread on import so server binds immediately.
Thread(target=init_services, daemon=True).start()

# Basic routes
@app.route("/")
def index():
    # make sure chat.html exists in templates/; otherwise show minimal message
    if os.path.exists(os.path.join(app.root_path, "templates", "chat.html")):
        return render_template("chat.html")
    return "<h2>Chat app</h2><p>Template missing. Check templates/chat.html</p>"

@app.route("/get", methods=["POST", "GET"])
def get_response():
    # simple API to accept form or JSON
    msg = request.values.get("msg") or (request.get_json(silent=True) or {}).get("msg")
    if not msg:
        return "No message provided", 400

    if not init_state.get("ready"):
        # return friendly message and tell them to check /health
        return jsonify({"error": "Backend not ready", "detail": init_state.get("error")}), 503

    try:
        qa = init_state.get("qa")
        result = qa({"query": msg})
        return jsonify({"result": result.get("result")})
    except Exception:
        tb = traceback.format_exc()
        logging.error("Query failed:\n" + tb)
        return jsonify({"error": "Server error", "detail": tb}), 500

@app.route("/health")
def health():
    return jsonify({
        "ready": init_state.get("ready"),
        "error": (init_state.get("error") or "")[:4000]  # truncated
    })

if _name_ == "_main_":
    # Ensure we bind to the Render-provided PORT
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Starting Flask on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
