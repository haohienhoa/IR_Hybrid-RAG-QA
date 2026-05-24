from src.data_loader import load_cranfield_docs
from src.retriever import build_hybrid_retriever
from src.agent import create_agent_graph

def main():
    print("="*60)
    print(" KHỞI ĐỘNG HỆ THỐNG")
    print("="*60)
    
    docs = load_cranfield_docs()
    hybrid_retriever = build_hybrid_retriever(docs)
    
    agent_app = create_agent_graph(hybrid_retriever)
    
    print("\n HỆ THỐNG SẴN SÀNG! (Gõ 'quit' để thoát)")
    while True:
        user_query = input("\n👤 Câu hỏi: ")
        if user_query.lower() == "quit":
            print("Tạm biệt!")
            break
            
        if not user_query.strip():
            continue
            
        inputs = {"query": user_query}
        result = agent_app.invoke(inputs)
        
        print("\n Trả lời:")
        print(result['answer'])
        print("-" * 60)

if __name__ == "__main__":
    main()