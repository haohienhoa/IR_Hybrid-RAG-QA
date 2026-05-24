from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.graph import StateGraph, END
from src.config import LLM_MODEL, HF_TOKEN

class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str

def create_agent_graph(retriever):
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        huggingfacehub_api_token=HF_TOKEN
    )
    llm = ChatHuggingFace(llm=llm_endpoint)

    def retrieve_node(state: AgentState):
        print(f"\n[Agent: Tìm kiếm] Đang đọc tài liệu cho: '{state['query']}'...")
        documents = retriever.invoke(state["query"])
        return {"context": documents}

    def generate_node(state: AgentState):
        print(f"[Agent: Tư duy ({LLM_MODEL})] Đang tổng hợp thông tin...")
        context_text = "\n\n".join([d.page_content for d in state["context"]])
        
        prompt = f"""Bạn là một chuyên gia kỹ thuật hàng không. Dựa vào các TÀI LIỆU dưới đây, hãy trả lời CÂU HỎI. 
        Nếu tài liệu tiếng Anh, hãy tự dịch và TRẢ LỜI BẰNG TIẾNG VIỆT thật tự nhiên.
        Tuyệt đối không bịa đặt thông tin.
        
        TÀI LIỆU:
        {context_text}
        
        CÂU HỎI: {state['query']}
        
        TRẢ LỜI:"""
        
        response = llm.invoke(prompt)
        return {"answer": response.content}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()