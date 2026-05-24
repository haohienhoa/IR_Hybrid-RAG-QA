import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

TOP_K_BM25 = 3
TOP_K_CHROMA = 3
CHROMA_PERSIST_DIR = "data/chroma_db"