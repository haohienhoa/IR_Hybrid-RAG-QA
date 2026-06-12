from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse 
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.data_loader import load_zac_legal_docs
from src.retriever import build_hybrid_retriever
from src.agent import create_multi_agent_system

rag_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    print("🚀 [STARTUP] Đang khởi tạo hệ thống...")
    try:
        docs = load_zac_legal_docs()
        hybrid_retriever = build_hybrid_retriever(docs)
        rag_chain = create_multi_agent_system(hybrid_retriever)
        print("✅ [STARTUP] Hệ thống RAG đã sẵn sàng nhận Request!")
    except Exception as e:
        print(f"❌ [LỖI KHỞI ĐỘNG]: {e}")
    yield

app = FastAPI(title="Vietnamese Legal RAG API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    status: str


@app.get("/", response_class=FileResponse)
def read_index():
    return "index.html" 

@app.post("/api/chat", response_model=QueryResponse)
async def chat_with_bot(request: QueryRequest):
    if not rag_chain:
        raise HTTPException(status_code=503, detail="Hệ thống AI đang khởi động...")
    
    try:
        answer = rag_chain.invoke(request.question)
        return QueryResponse(answer=answer, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi suy luận AI: {str(e)}")