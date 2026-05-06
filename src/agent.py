from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

from src.config import settings
from src.db import run_sql
from src.schema_indexer import get_vectorstore


class AgentState(TypedDict):
    user_query: str
    schema_context: str
    sql_query: str
    sql_result: list[dict]
    answer: str


def _llm():
    return HuggingFaceEndpoint(
        repo_id=settings.hf_model_id,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
    )


def retrieve_schema(state: AgentState) -> AgentState:
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(state["user_query"], k=8)
    context = "\n\n".join(d.page_content for d in docs)
    return {**state, "schema_context": context}


def generate_sql(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_template(
        """You are an expert SQL Server assistant.
Given user question and schema context, produce ONLY valid SQL query.
Do not include explanations, markdown fences, or comments.

User Question:
{question}

Schema Context:
{schema}
"""
    )
    llm = _llm()
    sql_query = llm.invoke(prompt.format(question=state["user_query"], schema=state["schema_context"]))
    return {**state, "sql_query": str(sql_query).strip()}


def execute_sql(state: AgentState) -> AgentState:
    rows = run_sql(state["sql_query"], limit=settings.max_sql_rows)
    return {**state, "sql_result": rows}


def draft_answer(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful data assistant.
Use user question and SQL results to produce a concise answer.
If results are empty, clearly say no records found.

User Question:
{question}

SQL Query:
{sql_query}

SQL Results:
{results}
"""
    )
    llm = _llm()
    answer = llm.invoke(
        prompt.format(
            question=state["user_query"],
            sql_query=state["sql_query"],
            results=state["sql_result"],
        )
    )
    return {**state, "answer": str(answer).strip()}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve_schema", retrieve_schema)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("draft_answer", draft_answer)

    graph.set_entry_point("retrieve_schema")
    graph.add_edge("retrieve_schema", "generate_sql")
    graph.add_edge("generate_sql", "execute_sql")
    graph.add_edge("execute_sql", "draft_answer")
    graph.add_edge("draft_answer", END)

    return graph.compile()


def ask_agent(question: str) -> dict:
    app = build_graph()
    initial_state: AgentState = {
        "user_query": question,
        "schema_context": "",
        "sql_query": "",
        "sql_result": [],
        "answer": "",
    }
    return app.invoke(initial_state)
