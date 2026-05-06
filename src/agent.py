from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from src.config import settings
from src.db import run_sql
from src.schema_indexer import get_vectorstore


class AgentState(TypedDict):
    user_query: str
    route: str
    schema_context: str
    sql_query: str
    sql_result: list[dict]
    answer: str


def _chat_llm():
    llm = HuggingFaceEndpoint(
        repo_id=settings.hf_model_id,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
    )
    return ChatHuggingFace(llm=llm)


def _as_text(output) -> str:
    return output.content.strip() if hasattr(output, "content") else str(output).strip()


def _strip_sql_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def decide_route(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_template(
        """Classify the user request into one label:
- DB_QUERY: user asks for records, tables, counts, filtering, SQL/data lookup from database
- GENERAL_CHAT: user asks general question, explanation, brainstorming, writing, or anything not requiring DB query

Return ONLY one token: DB_QUERY or GENERAL_CHAT.

User Question:
{question}
"""
    )
    llm = _chat_llm()
    result = _as_text(llm.invoke(prompt.format_messages(question=state["user_query"]))).upper()
    route = "DB_QUERY" if "DB_QUERY" in result else "GENERAL_CHAT"
    return {**state, "route": route}


def general_answer(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant.
Respond conversationally and clearly to the user.
If the user did not ask for database data, do not mention SQL tools.

User Question:
{question}
"""
    )
    llm = _chat_llm()
    answer = llm.invoke(prompt.format_messages(question=state["user_query"]))
    return {**state, "answer": _as_text(answer), "sql_query": "", "sql_result": []}


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
Domain guidance for STDBCOD:
- Source table for COD order issue tracking is `edm_cod_jsm_dly` (new COD order is loaded via the `issue_type` column).
- For creation logic (created filters, created ordering, new record detection), use `dice_ins_crt_dt` and/or `dice_ins_dt`.
- For updation logic (updated filters and updated ordering), use `dice_ins_upd_st`.
- Treat these `dice_` columns as authoritative over other date/timestamp fields when available in schema.

User Question:
{question}

Schema Context:
{schema}
"""
    )
    llm = _chat_llm()
    sql_query = llm.invoke(prompt.format_messages(question=state["user_query"], schema=state["schema_context"]))
    return {**state, "sql_query": _strip_sql_code_fence(_as_text(sql_query))}


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
    llm = _chat_llm()
    answer = llm.invoke(
        prompt.format_messages(
            question=state["user_query"],
            sql_query=state["sql_query"],
            results=state["sql_result"],
        )
    )
    return {**state, "answer": _as_text(answer)}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("decide_route", decide_route)
    graph.add_node("general_answer", general_answer)
    graph.add_node("retrieve_schema", retrieve_schema)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("draft_answer", draft_answer)

    graph.set_entry_point("decide_route")
    graph.add_conditional_edges(
        "decide_route",
        lambda state: state["route"],
        {
            "DB_QUERY": "retrieve_schema",
            "GENERAL_CHAT": "general_answer",
        },
    )
    graph.add_edge("general_answer", END)
    graph.add_edge("retrieve_schema", "generate_sql")
    graph.add_edge("generate_sql", "execute_sql")
    graph.add_edge("execute_sql", "draft_answer")
    graph.add_edge("draft_answer", END)

    return graph.compile()


def ask_agent(question: str) -> dict:
    app = build_graph()
    initial_state: AgentState = {
        "user_query": question,
        "route": "",
        "schema_context": "",
        "sql_query": "",
        "sql_result": [],
        "answer": "",
    }
    return app.invoke(initial_state)
