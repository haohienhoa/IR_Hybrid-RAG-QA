from datasets import load_dataset
from langchain_core.documents import Document

def load_zac_legal_docs():
    print(">> Đang kết nối tới Hugging Face...")
    print(">> Đang tải config 'corpus' từ dataset: minhnguyent546/zalo-ai-legal-text-retrieval-2021 ...")
    
    try:
        dataset = load_dataset("minhnguyent546/zalo-ai-legal-text-retrieval-2021", "corpus")
        
        split_name = 'corpus' if 'corpus' in dataset else 'train'
        corpus_data = dataset[split_name]
        
        docs = []
        
        for i, item in enumerate(corpus_data):
            doc_id = item.get('_id', str(i))
            title = item.get('title', '')
            text = item.get('text', '')
            
            if not text or not text.strip():
                continue
                
            enriched_text = f"[{title}] {text}" if title else text
                
            docs.append(Document(
                page_content=enriched_text,
                metadata={
                    "id": str(doc_id),
                    "title": str(title)
                }
            ))
            
        print(f">> Tải thành công {len(docs)} documents (Điều luật).")
        return docs
        
    except Exception as e:
        print(f"\n❌ Lỗi khi tải dữ liệu từ Hugging Face: {e}")
        print(">> Vui lòng kiểm tra lại kết nối mạng của bạn.")
        return []