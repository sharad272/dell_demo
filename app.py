import streamlit as st
import pandas as pd

from src.agent import ask_agent
from src.schema_indexer import build_or_refresh_vectorstore


st.set_page_config(page_title="SQL RAG Assistant", layout="wide")
st.title("Agentic SQL RAG Assistant")

with st.sidebar:
    st.subheader("Setup")
    if st.button("Refresh DB Schema Index"):
        try:
            with st.spinner("Fetching DB schema and rebuilding vector index..."):
                build_or_refresh_vectorstore()
            st.success("Schema index refreshed.")
        except Exception as exc:
            st.error(f"Failed to refresh schema index: {exc}")

question = st.text_area("Ask anything about records in your SQL Server database:", height=120)

if st.button("Ask Assistant", type="primary") and question.strip():
    try:
        with st.spinner("Running LangGraph agent..."):
            output = ask_agent(question)
    except Exception as exc:
        st.error(f"Assistant failed: {exc}")
        st.stop()

    st.subheader("Answer")
    st.write(output.get("answer", ""))

    st.subheader("Generated SQL")
    st.code(output.get("sql_query", ""), language="sql")

    st.subheader("SQL Results")
    rows = output.get("sql_result", [])
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No rows returned.")
