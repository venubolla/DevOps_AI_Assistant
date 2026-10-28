import json
import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 600

# Prior turns sent to the agent so follow-up questions keep their context.
HISTORY_TURNS = 10

st.set_page_config(page_title="DevOps AI Assistant", page_icon="🛠️")
st.title("DevOps AI Assistant")
st.caption("Ask questions answered from the DevOps knowledge base, with citations.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top K sources", min_value=1, max_value=10, value=5)
    service = st.text_input("Service filter (optional)")
    environment = st.text_input("Environment filter (optional)")
    document_type = st.text_input("Document type filter (optional)")
    st.caption(f"API: {API_URL}")
    if st.button("Clear chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            st.markdown(
                f"**{source['citation']} {source['title']}**  \n"
                f"`{source['source']}` — score {source['score']:.3f}"
            )


def build_payload(question, history):
    payload = {
        "question": question,
        "top_k": top_k,
        "history": [
            {"role": message["role"], "content": message["content"]} for message in history
        ],
    }
    for key, value in (
        ("service", service),
        ("environment", environment),
        ("document_type", document_type),
    ):
        if value.strip():
            payload[key] = value.strip()

    return payload


def stream_answer(question, history, status_placeholder):
    """Consume the NDJSON progress stream and return the final answer and sources."""

    with httpx.stream(
        "POST",
        f"{API_URL}/api/v1/agent/ask/stream",
        json=build_payload(question, history),
        timeout=REQUEST_TIMEOUT_SECONDS,
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line.strip():
                continue

            event = json.loads(line)

            if event["type"] == "status":
                status_placeholder.caption(f"{event['stage']}...")
            elif event["type"] == "error":
                return event["message"], []
            elif event["type"] == "result":
                return event["answer"], event["sources"]

    return "The agent did not return an answer.", []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources"))

if question := st.chat_input("Ask a DevOps question..."):
    history = st.session_state.messages[-HISTORY_TURNS:]

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()

        try:
            answer, sources = stream_answer(question, history, status_placeholder)
        except httpx.HTTPError as exc:
            answer, sources = f"Could not get an answer from the API: {exc}", []

        status_placeholder.empty()
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
