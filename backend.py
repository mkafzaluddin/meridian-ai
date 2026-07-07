import os
import uuid
from dotenv import load_dotenv

load_dotenv()

print("TOKEN LOADED:", os.getenv("MERIDIAN_TEST_TOKEN", "NOT FOUND")[:20])

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pypdf import PdfReader
import io

# Update import
# Update import
from tools import check_order_status, get_menu, cancel_order, get_my_orders, add_to_cart, place_order



app = FastAPI(title="Meridian AI")

# ─── Setup — runs ONCE when server starts ─────────────────────────────

search_tool = TavilySearch(max_results=5)

llm_map = {
    "openai/gpt-oss-20b": ChatGroq(model="openai/gpt-oss-20b"),
}

# Loaded lazily — only when PDF is uploaded
embeddings = None

def get_embeddings():
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

# RAG state
rag_store = {"vector_store": None, "filename": None, "num_chunks": 0}

# Conversation memory — session_id → list of message dicts
# Each dict: {"role": "user" | "assistant", "content": "..."}
conversation_store = {}

# ─── Schemas ──────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    model: str
    system_prompt: str
    allow_search: bool = True
    use_pdf: bool = False
    session_id: str = "default"  # identifies which conversation this belongs to


# ─── PDF Upload ───────────────────────────────────────────────────────


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    reader = PdfReader(io.BytesIO(contents))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    documents = []
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if not page_text:
            continue
        chunks = splitter.split_text(page_text)
        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"page": page_num, "source": file.filename},
                )
            )

    if not documents:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    # Uses the shared embeddings instance — no recreation
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)

    rag_store["vector_store"] = vector_store
    rag_store["filename"] = file.filename
    rag_store["num_chunks"] = len(documents)

    return {
        "message": "PDF processed successfully",
        "filename": file.filename,
        "pages": len(reader.pages),
        "chunks": len(documents),
    }


# ─── Chat ─────────────────────────────────────────────────────────────


@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    if request.model not in llm_map:
        raise HTTPException(status_code=400, detail="Invalid model selected")

    llm = llm_map[request.model]

    # ── RAG path ──────────────────────────────────────────────────────
    if request.use_pdf and rag_store["vector_store"] is not None:
        results = rag_store["vector_store"].similarity_search(request.message, k=3)

        if not results:
            return {"response": "No relevant content found in the uploaded PDF."}

        context = "\n\n".join(
            [f"[Page {r.metadata.get('page', '?')}]\n{r.page_content}" for r in results]
        )

        rag_system_prompt = f"""{request.system_prompt}

You have been given excerpts from a PDF. Use them to answer the question.
Always mention which page the information came from.
If the answer isn't in the excerpts, say "I couldn't find that in the document."

PDF Excerpts:
{context}
"""
        messages = [
            SystemMessage(content=rag_system_prompt),
            HumanMessage(content=request.message),
        ]
        response = llm.invoke(messages)

        # Save RAG turns to memory too
        _save_to_history(request.session_id, request.message, response.content)

        return {"response": response.content}

    # ── Agent path with memory ────────────────────────────────────────

    # Get this session's history — empty list if first message
    history = conversation_store.get(request.session_id, [])

    # Rebuild full message list: system + history + new message
    messages = [SystemMessage(content=request.system_prompt)]

    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=request.message))

    # Build agent and invoke with full history

    agent = create_agent(
        model=llm,
        tools=[search_tool, check_order_status, get_menu, cancel_order, get_my_orders, add_to_cart, place_order]
        if request.allow_search
        else [check_order_status, get_menu, cancel_order, get_my_orders, add_to_cart, place_order],
        system_prompt=request.system_prompt,
    )
    

    response = agent.invoke({"messages": messages})
    reply = response["messages"][-1].content

    # Save to memory
    _save_to_history(request.session_id, request.message, reply)

    return {"response": reply}


# ─── Helper — save a turn to conversation history ─────────────────────


def _save_to_history(session_id: str, user_msg: str, assistant_msg: str):
    if session_id not in conversation_store:
        conversation_store[session_id] = []

    conversation_store[session_id].append({"role": "user", "content": user_msg})
    conversation_store[session_id].append(
        {"role": "assistant", "content": assistant_msg}
    )

    # Keep only last 20 messages (10 turns) — prevents context overflow
    conversation_store[session_id] = conversation_store[session_id][-20:]


# ─── Clear conversation ───────────────────────────────────────────────


@app.delete("/conversation/{session_id}")
def clear_conversation(session_id: str):
    if session_id in conversation_store:
        del conversation_store[session_id]
    return {"message": "Conversation cleared"}


# ─── PDF status ───────────────────────────────────────────────────────


@app.get("/pdf-status")
def pdf_status():
    if rag_store["vector_store"] is None:
        return {"loaded": False}
    return {
        "loaded": True,
        "filename": rag_store["filename"],
        "chunks": rag_store["num_chunks"],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
