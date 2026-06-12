import os
import re
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from dotenv import load_dotenv

load_dotenv()

class CleanOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
        if match: return match.group(1).strip()
        return re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

class AgentState(TypedDict):
    question: str
    rewritten_query: str
    documents: str
    answer: str

def create_multi_agent_system(advanced_retriever):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.1,
    )

    def legal_analyst_agent(state: AgentState):
        print(f"\n[Legal Analyst] Đang phân tích câu hỏi: '{state['question']}'")
        
        prompt = PromptTemplate.from_template(
            """Bạn là chuyên gia phân tích pháp lý. Hãy tóm tắt câu hỏi kể chuyện của người dùng thành MỘT CÂU TRUY VẤN ngắn gọn, chứa các keyword pháp lý chuẩn mực để tìm kiếm trong cơ sở dữ liệu Luật.
            KHÔNG giải thích, KHÔNG trả lời, CHỈ IN RA câu truy vấn.
            Câu hỏi gốc: {question}
            Từ khóa truy vấn:"""
        )
        
        chain = prompt | llm | StrOutputParser()
        rewritten_query = chain.invoke({"question": state["question"]}).strip()
        
        print(f"[Legal Analyst] Đã trích xuất từ khóa: '{rewritten_query}'")
        return {"rewritten_query": rewritten_query}

    def paralegal_agent(state: AgentState):
        print("[Paralegal] Đang lục tìm kho dữ liệu và chạy Cross-Encoder Reranker...")
        
        docs = advanced_retriever.invoke(state["rewritten_query"])
        context = "\n\n".join([f"Tài liệu {i+1}: {d.page_content}" for i, d in enumerate(docs)])
        
        print(f"[Paralegal] Đã chốt lại {len(docs)} điều luật liên quan nhất gửi cho Luật sư.")
        return {"documents": context}

    def senior_lawyer_agent(state: AgentState):
        print("[Senior Lawyer] Đang tổng hợp hồ sơ và soạn thảo câu trả lời...")
        
        prompt = PromptTemplate.from_template(
            """Bạn là một Luật sư cấp cao tại Việt Nam. 
Nhiệm vụ của bạn là tư vấn pháp lý dựa trên các điều luật do Trợ lý cung cấp.

QUY TẮC BẮT BUỘC:
1. Trả lời hoàn toàn bằng TIẾNG VIỆT.
2. TUYỆT ĐỐI CHỈ DÙNG thông tin trong "TÀI LIỆU TỪ TRỢ LÝ". Không được dùng kiến thức bên ngoài.
3. Luôn trình bày tư duy logic vào thẻ <thought> trước khi đưa ra câu trả lời cuối cùng vào thẻ <answer>.

--- VÍ DỤ 1 (CÓ THÔNG TIN) ---
TÀI LIỆU TỪ TRỢ LÝ: 
Tài liệu 1: [Luật Giao thông đường bộ 2008] Điều 8. Các hành vi bị nghiêm cấm: 8. Điều khiển xe ô tô, máy kéo, xe máy chuyên dùng trên đường mà trong máu hoặc hơi thở có nồng độ cồn.
Câu hỏi của khách hàng: Tôi uống 1 lon bia rồi lái xe ô tô về thì có vi phạm luật không?
<thought>
Tài liệu 1 (Điều 8 Luật Giao thông đường bộ 2008) nghiêm cấm hành vi điều khiển xe ô tô khi có nồng độ cồn. Uống 1 lon bia chắc chắn sẽ tạo ra nồng độ cồn trong hơi thở/máu. Do đó, hành vi này là vi phạm pháp luật. Tôi sẽ kết luận và trích dẫn Điều 8.
</thought>
<answer>
Hành vi lái xe ô tô sau khi uống bia của bạn là vi phạm pháp luật.
Theo quy định tại khoản 8 Điều 8 Luật Giao thông đường bộ 2008, pháp luật nghiêm cấm hành vi điều khiển xe ô tô trên đường mà trong máu hoặc hơi thở có nồng độ cồn.
</answer>

--- VÍ DỤ 2 (KHÔNG CÓ THÔNG TIN / LẠC ĐỀ) ---
TÀI LIỆU TỪ TRỢ LÝ:
Tài liệu 1: [Luật Hôn nhân và gia đình 2014] Điều 8. Điều kiện kết hôn: 1. Nam, nữ kết hôn với nhau phải tuân theo các điều kiện...
Câu hỏi của khách hàng: Trộm cắp điện thoại trị giá 30 triệu thì đi tù mấy năm?
<thought>
Tài liệu được cung cấp chỉ nói về điều kiện kết hôn theo Luật Hôn nhân và gia đình. Không có bất kỳ thông tin nào liên quan đến hình phạt cho tội trộm cắp tài sản. Để đảm bảo tính chính xác của tư vấn pháp lý, tôi bắt buộc phải từ chối trả lời.
</thought>
<answer>
Dựa trên dữ liệu pháp lý hiện tại được cung cấp, tôi không tìm thấy quy định chính xác cho trường hợp này.
</answer>
-------------------------------

--- HỒ SƠ VỤ ÁN HIỆN TẠI ---
TÀI LIỆU TỪ TRỢ LÝ:
{documents}

---
Câu hỏi của khách hàng: {question}"""
        )
        
        chain = prompt | llm | CleanOutputParser()
        final_answer = chain.invoke({"documents": state["documents"], "question": state["question"]})
        
        print("[Hoàn tất] Câu trả lời đã sẵn sàng gửi về Client.")
        return {"answer": final_answer}

    workflow = StateGraph(AgentState)

    workflow.add_node("Legal_Analyst", legal_analyst_agent)
    workflow.add_node("Paralegal", paralegal_agent)
    workflow.add_node("Senior_Lawyer", senior_lawyer_agent)

    workflow.set_entry_point("Legal_Analyst")
    workflow.add_edge("Legal_Analyst", "Paralegal")
    workflow.add_edge("Paralegal", "Senior_Lawyer")
    workflow.add_edge("Senior_Lawyer", END)

    multi_agent_system = workflow.compile()
    
    class AgentWrapper:
        def invoke(self, question: str):
            result = multi_agent_system.invoke({"question": question})
            return result["answer"]

    return AgentWrapper()