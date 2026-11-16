
import os
import requests
import streamlit as st

API_URL = os.getenv(
    "ORBITDESK_API_URL",
    "http://127.0.0.1:8000"
)

st.set_page_config(
    page_title="OrbitDesk Support Agent",
    page_icon="💬",
    layout="centered"
)

st.title("OrbitDesk Support Agent")
st.caption("AI-powered support assistant using LangGraph and RAG.")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []


def get_agent_response(question):
    response = requests.post(
        f"{API_URL}/chat",
        json={"question": question},
        timeout=180
    )

    response.raise_for_status()
    return response.json()


# Sidebar
with st.sidebar:
    st.header("Connection")

    if st.button("Check API health"):
        try:
            response = requests.get(
                f"{API_URL}/health",
                timeout=5
            )
            response.raise_for_status()
            st.success("FastAPI connected successfully!")
        except requests.RequestException:
            st.error("Cannot connect to FastAPI.")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            result = message.get("result", {})

            if result:
                st.caption(
                    f"Classification: {result.get('classification')} "
                    f"| Confidence: {result.get('confidence', 0):.0%}"
                )

                if result.get("requires_escalation"):
                    st.warning("Human support is required.")

                if result.get("citations"):
                    with st.expander("View citations"):
                        for citation in result["citations"]:
                            st.markdown(
                                f"**{citation['source_id']}**"
                            )
                            st.write(citation["passage"])

                if result.get("reason"):
                    with st.expander("Decision details"):
                        st.write(result["reason"])


# Chat input
question = st.chat_input("Ask OrbitDesk a question...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("OrbitDesk is thinking..."):
            try:
                result = get_agent_response(question)
                answer = result.get(
                    "answer",
                    "No answer received."
                )

                st.markdown(answer)

                st.caption(
                    f"Classification: {result.get('classification')} "
                    f"| Confidence: {result.get('confidence', 0):.0%}"
                )

                if result.get("requires_escalation"):
                    st.warning("Human support is required.")

                if result.get("citations"):
                    with st.expander("View citations"):
                        for citation in result["citations"]:
                            st.markdown(
                                f"**{citation['source_id']}**"
                            )
                            st.write(citation["passage"])

                if result.get("reason"):
                    with st.expander("Decision details"):
                        st.write(result["reason"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "result": result
                })

            except requests.RequestException as error:
                error_message = (
                    f"Unable to reach the backend: {error}"
                )
                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
