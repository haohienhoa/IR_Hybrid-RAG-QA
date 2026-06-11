import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

LLM_MODEL = "google/gemma-2-2b-it"

EMBEDDING_MODEL = "keepitreal/vietnamese-sbert" 

TOP_K_BM25 = 4
TOP_K_CHROMA = 4
CHROMA_PERSIST_DIR = "data/chroma_db"