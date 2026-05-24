import ir_datasets
from langchain_core.documents import Document

def load_cranfield_docs():
    """Tải dữ liệu Cranfield và ép kiểu sang LangChain Document format"""
    print(">> Đang tải dataset Cranfield...")
    dataset = ir_datasets.load("cranfield")
    
    docs = []
    for d in dataset.docs_iter():
        docs.append(Document(
            page_content=d.text,
            metadata={"id": d.doc_id, "title": d.title}
        ))
    print(f">> Tải thành công {len(docs)} documents.")
    return docs