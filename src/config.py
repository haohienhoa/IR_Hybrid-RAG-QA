import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-1.5-flash"

# Cấu hình Retrieval
TOP_K_BM25 = 3
TOP_K_CHROMA = 3
CHROMA_PERSIST_DIR = "data/chroma_db"