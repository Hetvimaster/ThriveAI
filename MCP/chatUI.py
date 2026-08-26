# import streamlit as st
# from exaGraph import run_sync

# st.title("MCP Agent UI")

# query = st.text_input("Ask something:")

# if st.button("Run"):
#     response = run_sync(query)
#     st.write(response)

import streamlit as st
import uuid
from exaGraph2 import run_sync

st.set_page_config(
    page_title="ThriveAI",
    layout="wide"
)

# -----------------------------
# SESSION STATE
# -----------------------------
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_chat" not in st.session_state:
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat = chat_id
    st.session_state.conversations[chat_id] = {
        "title": "New Chat",
        "messages": []
    }

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:

    st.title("ThriveAI")

    if st.button("New Chat", use_container_width=True):
        chat_id = str(uuid.uuid4())

        st.session_state.conversations[chat_id] = {
            "title": "New Chat",
            "messages": []
        }

        st.session_state.current_chat = chat_id
        st.rerun()

    st.divider()

    st.subheader("Conversations")

    for chat_id, chat_data in st.session_state.conversations.items():

        if st.button(
            chat_data["title"],
            key=chat_id,
            use_container_width=True
        ):
            st.session_state.current_chat = chat_id
            st.rerun()

# -----------------------------
# CURRENT CHAT
# -----------------------------
current_chat = st.session_state.conversations[
    st.session_state.current_chat
]

# -----------------------------
# DISPLAY MESSAGES
# -----------------------------
for msg in current_chat["messages"]:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# CHAT INPUT
# -----------------------------
prompt = st.chat_input("Ask something...")

if prompt:

    if current_chat["title"] == "New Chat":
        current_chat["title"] = prompt[:30]

    current_chat["messages"].append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    response = run_sync(prompt,st.session_state.current_chat)

    current_chat["messages"].append({
        "role": "assistant",
        "content": response
    })

    with st.chat_message("assistant"):
        st.markdown(response)

    st.rerun()