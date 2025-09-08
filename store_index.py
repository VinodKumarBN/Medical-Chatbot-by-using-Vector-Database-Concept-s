from src.helper import chunks, local_emb
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os, hashlib, time

# Load env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medical-chatbot"

# Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Vector store with HuggingFace embeddings
vec_store = PineconeVectorStore(index=index, embedding=local_emb)

# ID generator
def make_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# Batch insert
BATCH_SIZE = 64
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i+BATCH_SIZE]
    ids = [make_id(c.page_content) for c in batch]
    vec_store.add_documents(batch, ids=ids)
    print(f"✅ Upserted batch {i}-{i+len(batch)-1}")
    time.sleep(0.5)

print("🎉 Local ingestion complete")




