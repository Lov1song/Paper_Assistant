import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

from src.load_all_paper import load_all_papers
from src.agent import SYSTEM_PROMPT, run_agent_turn
from src.tools import list_papers

load_dotenv()

# ====== 全局状态 ======
STATE = {
    "chunks": [],
    "vectors": None,
}

# session_id -> messages 历史
SESSION_STORE: dict[str, list[dict]] = {}

CLIENT = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


# ====== 启动/关闭时加载论文 ======
@asynccontextmanager
async def lifespan(app: FastAPI):
    chunks, vectors = load_all_papers()
    STATE["chunks"] = chunks
    STATE["vectors"] = vectors
    print(f"已加载 {len(chunks)} 个 chunks")
    yield
    # 关闭时无需清理


app = FastAPI(title="Paper Assistant API", lifespan=lifespan)


# ====== Pydantic 模型 ======
class ChatRequest(BaseModel):
    session_id: str
    question: str

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    steps: int  # messages 列表长度，间接反映经历了几轮工具调用


# ====== Endpoints ======

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    chunks = STATE["chunks"]
    vectors = STATE["vectors"]

    if not chunks:
        raise HTTPException(status_code=503, detail="论文索引尚未加载")

    # 取出或新建该 session 的 messages
    if req.session_id not in SESSION_STORE:
        SESSION_STORE[req.session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages = SESSION_STORE[req.session_id]
    messages.append({"role": "user", "content": req.question})

    answer, updated_messages = run_agent_turn(messages, chunks, vectors, CLIENT)
    SESSION_STORE[req.session_id] = updated_messages

    return ChatResponse(
        session_id=req.session_id,
        answer=answer,
        steps=len(updated_messages),
    )


@app.get("/papers")
def get_papers():
    chunks = STATE["chunks"]
    if not chunks:
        raise HTTPException(status_code=503, detail="论文索引尚未加载")
    return {"result": list_papers(chunks)}


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id in SESSION_STORE:
        del SESSION_STORE[session_id]
    return {"message": f"会话 {session_id} 已清除"}
