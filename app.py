# app.py
import os
import logging
import traceback
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Global state – no heavy objects yet
init_state = {
    "qa": None,
    "ready": False,
    "error": None,
}

# Config
INDEX_NAME = os.getenv("PINECONE_INDEX", "medical-chatbot")


def init_services():
    """Initialize embeddings, Pinecone retriever, and Cohere LLM lazily."""
    if init_state["qa"] is not None:  # already ready
        return init_state["qa"]

    try:
        logging.info("Init: loading environment...")
        from dotenv import load_dotenv
        load_dotenv()

        COHERE_API_KEY = os.getenv("COHERE_API_KEY")
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        if not COHERE_API_KEY or not PINECONE_API_KEY:
            raise RuntimeError("Missing COHERE_API_KEY or PINECONE_API_KEY")

        # Heavy imports happen only once
        from langchain_pinecone import PineconeVectorStore
        from langchain.chains import RetrievalQA
        from src.helper import local_emb  # your sentence-transformers embedding
        from src.prompt import qa_prompt
        from langchain_cohere import ChatCohere

        logging.info("Init: setting up Pinecone retriever...")
        vec_store = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=local_emb
        )
        retriever = vec_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        logging.info("Init: setting up Cohere...")
        cohere_chat = ChatCohere(
            cohere_api_key=COHERE_API_KEY,
            model=os.getenv("COHERE_MODEL", "command-a-03-2025"),
        )

        logging.info("Init: building RetrievalQA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=cohere_chat,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": qa_prompt},
        )

        # Save in global state
        init_state["qa"] = qa_chain
        init_state["ready"] = True
        logging.info("Init: backend services ready ✅")
        return qa_chain

    except Exception:
        tb = traceback.format_exc()
        logging.error("Init failed:\n" + tb)
        init_state["error"] = tb
        init_state["ready"] = False
        return None


@app.route("/")
def index():
    if os.path.exists(os.path.join(app.root_path, "templates", "chat.html")):
        return render_template("chat.html")
    return "<h2>Chat app</h2><p>Template missing.</p>"


@app.route("/get", methods=["POST", "GET"])
def get_response():
    msg = request.values.get("msg") or (request.get_json(silent=True) or {}).get("msg")
    if not msg:
        return "No message provided", 400

    qa = init_services()
    if not qa:
        return jsonify({"error": "Backend init failed", "detail": init_state.get("error")}), 503

    try:
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
        "error": (init_state.get("error") or "")[:4000]
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Starting Flask on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
