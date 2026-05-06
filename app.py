import streamlit as st
import pandas as pd

from src.agent import ask_agent
from src.schema_indexer import build_or_refresh_vectorstore


st.set_page_config(page_title="SQL RAG Assistant", layout="wide")
st.title("Agentic SQL Chat Assistant")

with st.sidebar:
    st.subheader("Setup")
    if st.button("Refresh DB Schema Index"):
        try:
            with st.spinner("Fetching DB schema and rebuilding vector index..."):
                build_or_refresh_vectorstore()
            st.success("Schema index refreshed.")
        except Exception as exc:
            st.error(f"Failed to refresh schema index: {exc}")
    st.caption("Memory/context retention is intentionally minimal for now.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("route") == "DB_QUERY":
            with st.expander("Generated SQL"):
                st.code(msg.get("sql_query", ""), language="sql")
            with st.expander("SQL Results"):
                rows = msg.get("sql_result", [])
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("No rows returned.")

question = st.chat_input("Ask anything (general or database related)...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.spinner("Running LangGraph agent..."):
            output = ask_agent(question)
    except Exception as exc:
        st.error(f"Assistant failed: {exc}")
        st.stop()

    assistant_msg = {
        "role": "assistant",
        "content": output.get("answer", ""),
        "route": output.get("route", ""),
        "sql_query": output.get("sql_query", ""),
        "sql_result": output.get("sql_result", []),
    }
    st.session_state.messages.append(assistant_msg)

    with st.chat_message("assistant"):
        st.markdown(assistant_msg["content"])
        if assistant_msg.get("route") == "DB_QUERY":
            with st.expander("Generated SQL"):
                st.code(assistant_msg.get("sql_query", ""), language="sql")
            with st.expander("SQL Results"):
                rows = assistant_msg.get("sql_result", [])
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("No rows returned.")
