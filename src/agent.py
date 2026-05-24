from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from src.config import LLM_MODEL

# 1. Định nghĩa trạng thái của Agent
class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str

def create_agent_graph(retriever):
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)

    # Node 1: Tìm kiếm tài liệu
    def retrieve_node(state: AgentState):
        print(f"\n[Agent: Tìm kiếm] Đang tìm ngữ cảnh cho câu hỏi: '{state['query']}'...")
        documents = retriever.invoke(state["query"])
        return {"context": documents}

    # Node 2: Sinh câu trả lời
    def generate_node(state: AgentState):
        print("[Agent: Tư duy] Đang tổng hợp thông tin để trả lời...")
        context_text = "\n\n".join([d.page_content for d in state["context"]])
        
        prompt = f"""Bạn là một trợ lý ảo thông minh cho ngành kỹ thuật hàng không. 
        Sử dụng BẮT BUỘC các tài liệu dưới đây để trả lời câu hỏi. Nếu tài liệu không có thông tin, hãy nói "Tôi không tìm thấy thông tin trong dữ liệu".
        
        TÀI LIỆU:
        {context_text}
        
        CÂU HỎI: {state['query']}
        
        TRẢ LỜI CỦA BẠN:"""
        
        response = llm.invoke(prompt)
        return {"answer": response.content}

    # Khởi tạo Graph
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()