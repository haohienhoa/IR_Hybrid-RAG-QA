import re
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import BaseOutputParser
from dotenv import load_dotenv

load_dotenv()

class CleanOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
        if match: return match.group(1).strip()
        return re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

def create_rag_chain(retriever):

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile", 
        temperature=0.1,
    )

    prompt_template = """Dựa vào NGỮ CẢNH dưới đây, hãy trả lời câu hỏi của người dùng bằng TIẾNG VIỆT.
Nếu không có thông tin, hãy trả lời: "Dựa trên dữ liệu pháp lý hiện tại, tôi không tìm thấy quy định cho trường hợp này."
Vui lòng trình bày theo format:
<thought> (Suy luận của bạn) </thought>
<answer> (Câu trả lời và trích dẫn luật) </answer>

--- NGỮ CẢNH ---
{context}

---
Câu hỏi: {query}"""

    prompt = PromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        print(f"\n[Retriever] Đã lấy {len(docs)} điều luật. Đang gọi Llama-3 qua Groq API (Siêu tốc)...")
        return "\n\n".join([f"Doc {i+1}: {d.page_content}" for i, d in enumerate(docs)])

    return ({"context": retriever | format_docs, "query": RunnablePassthrough()} | prompt | llm | CleanOutputParser())