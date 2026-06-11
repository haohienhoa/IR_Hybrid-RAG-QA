import os
from tqdm import tqdm # Import thư viện thanh tiến trình
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pyvi import ViTokenizer
from src.config import TOP_K_BM25, TOP_K_CHROMA, CHROMA_PERSIST_DIR, EMBEDDING_MODEL

def vietnamese_tokenize(text):
    """Hàm tách từ tiếng Việt cho BM25"""
    return ViTokenizer.tokenize(text).split()

def build_hybrid_retriever(docs):
    print(f">> Đang load Vietnamese Embedding Model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(">> Đang thiết lập BM25 Retriever (với Vietnamese Tokenizer)...")

    bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=vietnamese_tokenize)
    bm25_retriever.k = TOP_K_BM25

    print(">> Đang thiết lập Chroma Vector Database...")

    if os.path.exists(CHROMA_PERSIST_DIR) and len(os.listdir(CHROMA_PERSIST_DIR)) > 0:
        print("   -> Đã tìm thấy cơ sở dữ liệu Vector cũ. Đang load siêu tốc từ ổ cứng...")
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR, 
            embedding_function=embeddings
        )
    else:
        print(f"   -> Lần chạy đầu: Đang Embedding {len(docs)} điều luật...")
        
        vectorstore = Chroma(
            embedding_function=embeddings, 
            persist_directory=CHROMA_PERSIST_DIR
        )
        
        batch_size = 500
        
        for i in tqdm(range(0, len(docs), batch_size), desc="Tiến trình Embedding", unit="lô"):
            batch = docs[i : i + batch_size]
            vectorstore.add_documents(batch)
            
        print("   -> Embedding hoàn tất và đã lưu xuống ổ cứng!")

    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_CHROMA})

    print(">> Kết hợp BM25 + Chroma thành Hybrid Retriever...")
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.6, 0.4] 
    )
    
    return hybrid_retriever