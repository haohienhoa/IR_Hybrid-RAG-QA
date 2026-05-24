from src.data_loader import load_cranfield_docs
from src.retriever import build_hybrid_retriever
from src.agent import create_agent_graph

def main():
    print("="*50)
    print(" KHỞI ĐỘNG HỆ THỐNG AGENTIC RAG (FAISS + BM25) ")
    print("="*50)
    
    # 1. Pipeline thiết lập
    docs = load_cranfield_docs()
    hybrid_retriever = build_hybrid_retriever(docs)
    agent_app = create_agent_graph(hybrid_retriever)
    
    # 2. Vòng lặp tương tác (Chat interface)
    print("\n HỆ THỐNG ĐÃ SẴN SÀNG! (Gõ 'exit' hoặc 'quit' để thoát)")
    while True:
        user_query = input("\n👤 Câu hỏi của bạn: ")
        if user_query.lower() in ['exit', 'quit']:
            print("Tạm biệt!")
            break
            
        if not user_query.strip():
            continue
            
        inputs = {"query": user_query}
        result = agent_app.invoke(inputs)
        
        print("\n Trả lời:")
        print(result['answer'])
        print("-" * 50)

if __name__ == "__main__":
    main()