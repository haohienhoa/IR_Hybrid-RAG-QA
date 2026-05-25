from src.data_loader import load_cranfield_docs
from src.retriever import build_hybrid_retriever
from src.agent import create_rag_chain  

def main():
    print("="*60)
    print(" KHỞI ĐỘNG HỆ THỐNG")
    print("="*60)
    
    docs = load_cranfield_docs()
    hybrid_retriever = build_hybrid_retriever(docs)
    
    rag_chain = create_rag_chain(hybrid_retriever)
    
    print("\n HỆ THỐNG ĐÃ SẴN SÀNG! (Gõ 'quit' để thoát)")
    while True:
        user_query = input("\nCâu hỏi: ")
        if user_query.lower() == "quit":
            print("Tạm biệt!")
            break
            
        if not user_query.strip():
            continue
            
        print("\nĐang suy luận...")
        result = rag_chain.invoke(user_query)
        
        print("\n" + result)
        print("-" * 60)

if __name__ == "__main__":
    main()