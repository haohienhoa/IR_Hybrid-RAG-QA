import os
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import TOP_K_BM25, TOP_K_CHROMA, CHROMA_PERSIST_DIR, EMBEDDING_MODEL

def build_hybrid_retriever(docs):
    print(f">> Đang load Local Embedding Model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(">> Đang thiết lập BM25 Retriever...")
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = TOP_K_BM25

    print(">> Đang thiết lập Chroma Vector Database...")
    if os.path.exists(CHROMA_PERSIST_DIR):
        print("   -> Tìm thấy DB cũ. Đang load từ local...")
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR, 
            embedding_function=embeddings
        )
    else:
        print("   -> Lần chạy đầu: Đang Embedding dữ liệu và lưu xuống ổ cứng...")
        vectorstore = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings, 
            persist_directory=CHROMA_PERSIST_DIR
        )

    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_CHROMA})

    print(">> Kết hợp BM25 + Chroma thành Hybrid Retriever...")
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )
    
    return hybrid_retriever