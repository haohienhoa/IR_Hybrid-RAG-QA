import os
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from src.config import TOP_K_BM25, TOP_K_CHROMA, CHROMA_PERSIST_DIR, EMBEDDING_MODEL

def build_hybrid_retriever(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # 1. Khởi tạo BM25 Retriever (Tìm kiếm theo Từ khóa)
    print(">> Đang thiết lập BM25 Retriever...")
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = TOP_K_BM25

    # 2. Khởi tạo Chroma Vector Database (Tìm kiếm theo Ngữ nghĩa)
    print(">> Đang thiết lập Chroma Vector Database...")
    if os.path.exists(CHROMA_PERSIST_DIR):
        print("   -> Tìm thấy Data cũ. Đang load từ local...")
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR, 
            embedding_function=embeddings
        )
    else:
        print("   -> Lần chạy đầu tiên: Đang Embedding dữ liệu và lưu xuống ổ cứng...")
        vectorstore = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings, 
            persist_directory=CHROMA_PERSIST_DIR
        )

    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_CHROMA})

    # 3. Kết hợp thành Hybrid Retriever
    print(">> Kết hợp BM25 + Chroma thành Hybrid Retriever...")
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]  # Chia đều trọng số 50-50
    )
    
    return hybrid_retriever