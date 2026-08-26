import streamlit as st
import hashlib
import asyncio
from exaGraph3 import run_sync
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="ThriveAI",
    layout="wide"
)

# =============================================================================
# MEMORY ARCHITECTURE OVERVIEW
# =============================================================================
#
# 1. SHORT-TERM (within one graph run)
#    → LangGraph State object
#    → Lives only during a single ainvoke() call
#
# 2. LONG-TERM CONVERSATION HISTORY (per chat thread)
#    → AsyncPostgresSaver (checkpointer)
#    → Table: checkpoints
#    → Keyed by thread_id e.g. "a3f9b2c1_0"
#    → Survives forever, loaded automatically by LangGraph on every run
#
# 3. WITHIN-THREAD SUMMARIZATION (token management)
#    → summarize_node in exaGraph.py
#    → Triggers after 40 messages, trims history, keeps rolling summary
#    → Prevents hitting LLM token limits on long conversations
#
# 4. CROSS-THREAD SEMANTIC MEMORY (user facts across all chats)
#    → AsyncPostgresStore (store)
#    → Table: store
#    → Keyed by ("user_memory", user_id) — same for ALL threads of a user
#    → memory_node extracts facts after every answer and writes here
#    → answer_node reads this before every response to personalize
#    → Example: user_id "a3f9b2c1" always reads the same fact bucket
#      whether they're in chat _0, _1, or a brand new chat next week
#
# =============================================================================


# =============================================================================
# HELPERS
# =============================================================================

def get_stable_user_id(username: str) -> str:
    """
    Converts username to a stable 16-char ID.
    This is used as BOTH:
    - The base for thread_ids: "a3f9b2c1_0", "a3f9b2c1_1"
    - The user_id for cross-thread memory: ("user_memory", "a3f9b2c1")
    """
    return hashlib.sha256(username.strip().lower().encode()).hexdigest()[:16]


def load_thread_messages(thread_id: str) -> list:
    """Fetch display messages for one thread from Postgres checkpointer."""
    async def _fetch():
        DB_URI = os.getenv("DATABASE_URL")
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint_tuple = await checkpointer.aget_tuple(config)

            if checkpoint_tuple is None:
                return []

            channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
            raw_messages = channel_values.get("messages", [])

            display_messages = []
            for msg in raw_messages:
                if isinstance(msg, HumanMessage):
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    display_messages.append({"role": "user", "content": content})
                elif isinstance(msg, AIMessage):
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    display_messages.append({"role": "assistant", "content": content})

            return display_messages

    return asyncio.run(_fetch())


def load_all_user_threads(user_id: str) -> dict:
    """
    Scan checkpoints table for all threads belonging to this user.
    Loads and returns full message history for each thread.
    """
    async def _fetch():
        DB_URI = os.getenv("DATABASE_URL")
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            await checkpointer.setup()
            conn = checkpointer.conn
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT DISTINCT thread_id
                    FROM checkpoints
                    WHERE thread_id LIKE %s
                    ORDER BY thread_id ASC
                    """,
                    (f"{user_id}_%",)
                )
                rows = await cur.fetchall()
            return [row["thread_id"] for row in rows]

    try:
        thread_ids = asyncio.run(_fetch())
    except Exception:
        return {}

    conversations = {}
    for thread_id in thread_ids:
        messages = load_thread_messages(thread_id)
        first_human = next(
            (m["content"] for m in messages if m["role"] == "user"),
            "New Chat"
        )
        conversations[thread_id] = {
            "title": first_human[:30],
            "messages": messages
        }

    return conversations


def get_chat_counter(user_id: str, conversations: dict) -> int:
    max_index = -1
    for thread_id in conversations:
        try:
            suffix = int(thread_id.replace(f"{user_id}_", ""))
            max_index = max(max_index, suffix)
        except ValueError:
            pass
    return max_index + 1


# =============================================================================
# LOGIN
# =============================================================================

def show_login():
    st.title("ThriveAI")
    st.markdown("### Enter your username to continue")
    st.markdown(
        "Your conversation history is saved. "
        "Use the same username to pick up where you left off."
    )

    username = st.text_input("Username", placeholder="e.g. yash")

    if st.button("Continue", use_container_width=True):
        if username.strip():
            user_id = get_stable_user_id(username.strip())

            st.session_state.username = username.strip()
            st.session_state.user_id = user_id  # stable ID used for memory + threads

            with st.spinner("Loading your conversations..."):
                conversations = load_all_user_threads(user_id)

            if conversations:
                st.session_state.conversations = conversations
                st.session_state.current_chat = list(conversations.keys())[-1]
                st.session_state.chat_counter = get_chat_counter(user_id, conversations)
            else:
                first_thread_id = f"{user_id}_0"
                st.session_state.conversations = {
                    first_thread_id: {"title": "New Chat", "messages": []}
                }
                st.session_state.current_chat = first_thread_id
                st.session_state.chat_counter = 0

            st.rerun()
        else:
            st.error("Please enter a username.")


# =============================================================================
# CHAT UI
# =============================================================================

def init_user_session():
    if "current_chat" not in st.session_state:
        first = list(st.session_state.conversations.keys())[0]
        st.session_state.current_chat = first


def create_new_chat():
    st.session_state.chat_counter += 1
    new_thread_id = f"{st.session_state.user_id}_{st.session_state.chat_counter}"
    st.session_state.conversations[new_thread_id] = {
        "title": "New Chat",
        "messages": []
    }
    st.session_state.current_chat = new_thread_id


def show_chat_ui():
    init_user_session()

    with st.sidebar:
        st.title("ThriveAI")
        st.markdown(f"👤 **{st.session_state.username}**")

        if st.button("Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.divider()

        if st.button("+ New Chat", use_container_width=True):
            create_new_chat()
            st.rerun()

        st.divider()
        st.subheader("Conversations")

        for thread_id, chat_data in st.session_state.conversations.items():
            is_active = thread_id == st.session_state.current_chat
            label = f"**{chat_data['title']}**" if is_active else chat_data["title"]
            if st.button(label, key=thread_id, use_container_width=True):
                st.session_state.current_chat = thread_id
                st.rerun()

    current_thread_id = st.session_state.current_chat
    current_chat = st.session_state.conversations[current_thread_id]

    for msg in current_chat["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask something...")

    if prompt:
        if current_chat["title"] == "New Chat":
            current_chat["title"] = prompt[:30]

        current_chat["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            # Pass both thread_id (for this chat's history)
            # and user_id (for cross-thread semantic memory)
            response = run_sync(
                query=prompt,
                thread_id=current_thread_id,
                user_id=st.session_state.user_id
            )

        current_chat["messages"].append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)

        st.rerun()


# =============================================================================
# ENTRY POINT
# =============================================================================

if "username" not in st.session_state:
    show_login()
else:
    show_chat_ui()
