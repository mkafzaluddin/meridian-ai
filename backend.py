import os
import uuid
from dotenv import load_dotenv

load_dotenv()

print("TOKEN LOADED:", os.getenv("MERIDIAN_TEST_TOKEN", "NOT FOUND")[:20])

from fastapi import FastAPI, HTTPException
# from fastapi import UploadFile, File  # PDF feature — disabled for deployment
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from fastapi.middleware.cors import CORSMiddleware

# from langchain_text_splitters import RecursiveCharacterTextSplitter  # PDF feature
# from langchain_core.vectorstores import InMemoryVectorStore  # PDF feature
# from langchain_core.documents import Document  # PDF feature
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from pypdf import PdfReader  # PDF feature
# import io  # PDF feature

from tools import (
    check_order_status,
    get_menu,
    cancel_order,
    get_my_orders,
    add_to_cart,
    place_order,
)

app = FastAPI(title="Meridian AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, replace * with your actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ─── Setup ────────────────────────────────────────────────────────────
search_tool = TavilySearch(max_results=5)

llm_map = {
    "openai/gpt-oss-20b": ChatGroq(model="openai/gpt-oss-20b"),
}

# PDF feature — disabled for deployment (re-enable when upgrading plan)
# embeddings = None
# def get_embeddings():
#     global embeddings
#     if embeddings is None:
#         from langchain_huggingface import HuggingFaceEmbeddings
#         embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     return embeddings
# rag_store = {"vector_store": None, "filename": None, "num_chunks": 0}

# Conversation memory
conversation_store = {}

# ─── Schema ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    model: str
    system_prompt: str
    allow_search: bool = True
    # use_pdf: bool = False  # PDF feature — disabled
    session_id: str = "default"

# ─── PDF endpoints — disabled for deployment ──────────────────────────
# @app.post("/upload-pdf")
# async def upload_pdf(file: UploadFile = File(...)):
#     ... re-enable when upgrading Render plan ...
#
# @app.get("/pdf-status")
# def pdf_status():
#     ... re-enable when upgrading Render plan ...

# ─── Chat endpoint ─────────────────────────────────────────────────────
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    if request.model not in llm_map:
        raise HTTPException(status_code=400, detail="Invalid model selected")

    llm = llm_map[request.model]
    history = conversation_store.get(request.session_id, [])

    messages = [SystemMessage(content=request.system_prompt)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=request.message))

    agent = create_agent(
        model=llm,
        tools=[search_tool, check_order_status, get_menu, cancel_order, get_my_orders, add_to_cart, place_order]
        if request.allow_search
        else [check_order_status, get_menu, cancel_order, get_my_orders, add_to_cart, place_order],
        system_prompt=request.system_prompt,
    )

    response = agent.invoke({"messages": messages})
    reply = response["messages"][-1].content

    _save_to_history(request.session_id, request.message, reply)
    return {"response": reply}

# ─── Helper ────────────────────────────────────────────────────────────
def _save_to_history(session_id: str, user_msg: str, assistant_msg: str):
    if session_id not in conversation_store:
        conversation_store[session_id] = []
    conversation_store[session_id].append({"role": "user", "content": user_msg})
    conversation_store[session_id].append({"role": "assistant", "content": assistant_msg})
    conversation_store[session_id] = conversation_store[session_id][-20:]

# ─── Clear conversation ────────────────────────────────────────────────
@app.delete("/conversation/{session_id}")
def clear_conversation(session_id: str):
    if session_id in conversation_store:
        del conversation_store[session_id]
    return {"message": "Conversation cleared"}

# ─── Health check ──────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "Meridian AI is running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)