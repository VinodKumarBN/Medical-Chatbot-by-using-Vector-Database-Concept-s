# app.py
import os, traceback, json
from flask import Flask, render_template, request

# load local .env for local testing only (won't be used on Render)
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

init_error = None
qa = None  # will hold the RetrievalQA chain if init succeeds

def init_services():
    # check required env vars
    required = ["COHERE_API_KEY", "PINECONE_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

    # lazy imports (avoid import-time failures)
    from src.helper import ChatCohere, local_emb
    from langchain_pinecone import PineconeVectorStore
    from langchain.chains import RetrievalQA
    from src.prompt import qa_prompt

    index_name = os.getenv("PINECONE_INDEX", "medical-chatbot")

    # create vector store & retriever
    vec_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=local_emb
    )
    retriever = vec_store.as_retriever(search_type="similarity", search_kwargs={"k":4})

    # instantiate LLM (Cohere wrapper in your repo)
    cohere_chat = ChatCohere(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        model=os.getenv("COHERE_MODEL", "command-a-03-2025")
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=cohere_chat,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt}
    )
    return qa_chain

# Attempt to initialize services but don't crash on failure
try:
    qa = init_services()
except Exception:
    init_error = traceback.format_exc()
    qa = None

@app.route("/")
def index():
    if qa is None:
        # show a simple page explaining service not ready (avoid exposing secrets)
        return (
            "<h2>Service not ready</h2>"
            "<p>The backend services failed to initialize. Check logs for details.</p>"
            "<pre>{}</pre>".format(init_error[:1000]), 500
        )
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def get_response():
    if qa is None:
        return "Service not available", 503
    msg = request.form.get("msg", "")
    result = qa({"query": msg})
    return str(result.get("result", ""))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


