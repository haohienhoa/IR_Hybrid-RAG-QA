from src.data_loader import load_zac_legal_docs
from src.retriever import build_hybrid_retriever
from src.agent import create_rag_chain  

def main():
    print("="*70)
    print(" KHỞI ĐỘNG HỆ THỐNG TRUY XUẤT PHÁP LUẬT VN (ZALO ZAC HYBRID RAG)")
    print("="*70)
    
    docs = load_zac_legal_docs()
    
    hybrid_retriever = build_hybrid_retriever(docs)
    
    rag_chain = create_rag_chain(hybrid_retriever)
    
    print("\n" + "="*70)
    print(" HỆ THỐNG ĐÃ SẴN SÀNG! ".center(70, '*'))
    print(" (Gõ 'quit' hoặc 'exit' để thoát) ".center(70, ' '))
    print("="*70)
    
    while True:
        user_query = input("\n👤 Nhập tình huống/câu hỏi pháp lý: ")
        
        # Điều kiện thoát
        if user_query.lower() in ["quit", "exit"]:
            print("\n👋 Tạm biệt! Hẹn gặp lại.")
            break
            
        if not user_query.strip():
            continue
            
        print("\n🤖 Đang tra cứu luật và suy luận...")
        
        try:
            result = rag_chain.invoke(user_query)
            
            print("\n" + result)
            print("-" * 70)
        except Exception as e:
            print(f"\n❌ Đã xảy ra lỗi trong quá trình xử lý: {e}")
            print("Vui lòng thử lại với một câu hỏi khác.")

if __name__ == "__main__":
    main()