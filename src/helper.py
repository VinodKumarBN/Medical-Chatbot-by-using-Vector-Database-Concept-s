from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings, ChatCohere
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables (COHERE + PINECONE)
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
INDEX_NAME = "medical-chatbot"

# --- Fix path for Render/Linux ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # points to /src
DATA_PATH = os.path.join(BASE_DIR, "..", "data")       # project_root/data

# 1) Load & split PDF files
loader = DirectoryLoader(
    DATA_PATH,
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=40)
chunks = splitter.split_documents(docs)

# 2) Embeddings
local_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# 3) Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)
dim = len(local_emb.embed_query("test"))

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)
vec_store = PineconeVectorStore(index=index, embedding=local_emb)








