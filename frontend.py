import streamlit as st
import requests
import uuid
import os

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Meridian Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Meridian Eats AI Chatbot")

# ── Session state — persists across Streamlit reruns ──────────────────
# Streamlit reruns the whole script on every interaction.
# session_state is the only way to keep data alive between reruns.

if "session_id" not in st.session_state:
    # Unique ID per browser tab — this is how backend knows whose memory to use
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    # Local copy of chat for display — backend has its own copy for LLM context
    st.session_state.chat_history = []

# ─── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # gpt-4o-mini removed — no OpenAI credits
    model = st.selectbox("Model", options=["openai/gpt-oss-20b"])

    system_prompt = st.text_area(
        "System Prompt",
        value="""You are a friendly ordering assistant for Meridian Eats food delivery.

MENU:
- If the user asks for "full menu", "whole menu", "all items", or "everything" 
  → immediately call get_menu("all") and display ALL items grouped by category
- If the user asks for a specific category like "show me salads" 
  → call get_menu("Salad") and show just that category
- If the user says "I want to order something" with no category 
  → list the categories as options, don't call get_menu yet
- NEVER ask "which category?" when the user already said "full menu" or "all"
- Always show item names, IDs, and prices — never hide or summarize them

ORDERING RULES — follow these steps IN ORDER, never skip:
1. Show the menu for the requested category using get_menu
2. Ask which item the customer wants
3. Ask for the quantity — NEVER assume quantity, always ask explicitly
4. Ask if they have a promo/coupon code (we don't process it yet, just acknowledge)
5. Ask for delivery details: street, city, state, ZIP code, phone number
6. Summarize the full order and ask for confirmation
7. Only after confirmation — call add_to_cart then place_order

IMPORTANT:
- Never skip the quantity question
- Never place an order without explicit customer confirmation
- If the customer says just a food name with no quantity, ask "How many would you like?"
- Be concise and friendly""",
        height=250,
    )

    allow_search = st.checkbox("🌐 Allow Web Search", value=True)

    st.divider()

    # Clear conversation
    if st.button("🗑️ Clear Conversation"):
        # Tell backend to delete this session's history
        requests.delete(f"{BACKEND}/conversation/{st.session_state.session_id}")
        # Reset local display history
        st.session_state.chat_history = []
        # New session ID so history is truly fresh
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

    st.divider()

    # PDF upload
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        if st.button("Process PDF"):
            with st.spinner("Chunking and embedding..."):
                response = requests.post(
                    f"{BACKEND}/upload-pdf",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                )
            if response.status_code == 200:
                result = response.json()
                st.success(
                    f"✅ {result['filename']}\n"
                    f"{result['pages']} pages → {result['chunks']} chunks"
                )
            else:
                st.error(f"Failed: {response.text}")

    # Always show what PDF is currently loaded
    try:
        status = requests.get(f"{BACKEND}/pdf-status").json()
        if status["loaded"]:
            st.info(f"📄 **{status['filename']}** ({status['chunks']} chunks)")
        else:
            st.caption("No PDF loaded")
    except:
        st.caption("Backend offline")

# ─── Main chat area ───────────────────────────────────────────────────

# Query mode — above chat so user sets it before typing
query_mode = st.radio(
    "Query against:",
    options=["Agent (web search / general)", "Uploaded PDF"],
    horizontal=True,
)
use_pdf = query_mode == "Uploaded PDF"

st.divider()

# Display full conversation history as chat bubbles
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# Chat input — stays pinned to bottom, submits on Enter
if query := st.chat_input("Ask anything..."):
    # Show user message immediately — don't wait for backend
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Call backend and stream response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            payload = {
                "message": query,
                "model": model,
                "system_prompt": system_prompt,
                "allow_search": allow_search,
                "use_pdf": use_pdf,
                "session_id": st.session_state.session_id,
            }
            response = requests.post(f"{BACKEND}/chat", json=payload)

        if response.status_code == 200:
            reply = response.json()["response"]
            st.markdown(reply)

            # Show source tag
            if use_pdf:
                st.caption("📄 Sourced from PDF")
            elif allow_search:
                st.caption("🌐 May include web search")

            st.session_state.chat_history.append(
                {"role": "assistant", "content": reply}
            )
        else:
            error_msg = f"Error {response.status_code}: {response.text}"
            st.error(error_msg)
