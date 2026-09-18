import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Fitness & Wellness Assistant", page_icon="🏋️")
st.title("🏋️ Fitness & Wellness RAG Assistant")

# --- Session state initialization ---
# Streamlit reruns this whole script on every interaction, so anything that
# must survive across reruns (session_id, chat history) has to live in
# st.session_state, not as a plain local variable.
if "session_id" not in st.session_state:
    try:
        resp = requests.post(f"{API_BASE_URL}/api/sessions", timeout=10)
        resp.raise_for_status()
        st.session_state.session_id = resp.json()["session_id"]
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the API to start a session: {e}")
        st.stop()  # halts this script run; nothing below executes

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"model", "content": "..."}

# --- Render existing conversation ---
for msg in st.session_state.messages:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

# --- Handle new input ---
question = st.chat_input("Ask about fitness, nutrition, or wellness...")

if question:
    # Show the user's message immediately
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Call the API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/api/sessions/{st.session_state.session_id}/messages",
                    json={"question": question},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]
                chunks = data.get("chunks", [])
                retrieval_time = data.get("retrieval_time", 0)
                generation_time = data.get("generation_time", 0)

                st.markdown(answer)

                with st.expander(f"Sources & timing (retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)"):
                    for chunk in chunks:
                        # chunks are raw tuples: (chunk_id, chapter, section, source, chunk_text, word_count, similarity)
                        st.markdown(f"**{chunk[1]} — {chunk[2] or 'N/A'}** (similarity: {chunk[6]:.2f})")
                        st.caption(chunk[4][:300] + "...")

            except requests.exceptions.RequestException as e:
                answer = f"⚠️ Error reaching the assistant: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "model", "content": answer})

# --- Sidebar: session list + switch + new session ---
with st.sidebar:
    st.subheader("Sessions")

    if st.button("➕ New Session"):
        for key in ("session_id", "messages"):
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()

    try:
        resp = requests.get(f"{API_BASE_URL}/api/session_ids", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        sessions_list = data["sessions"]  # list of {"session_id": ..., "first_message": ...}
    except requests.exceptions.RequestException as e:
        sessions_list = []
        st.caption(f"Couldn't load session list: {e}")

    PREVIEW_LENGTH = 40

    for session in sessions_list:
        sid = session["session_id"]
        first_msg = session["first_message"]

        if not first_msg:
            continue

        preview = first_msg if len(first_msg) <= PREVIEW_LENGTH else first_msg[:PREVIEW_LENGTH].rstrip() + "..."

        is_current = sid == st.session_state.get("session_id")
        if st.button(("👉 " if is_current else "") + preview, key=f"session_{sid}"):
            if not is_current:
                try:
                    resp = requests.get(f"{API_BASE_URL}/api/sessions/{sid}", timeout=10)
                    resp.raise_for_status()
                    st.session_state.session_id = sid
                    st.session_state.messages = resp.json()["messages"]
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not load session: {e}")

    st.divider()
    st.caption(f"Current session: `{st.session_state.get('session_id', 'none')}`")