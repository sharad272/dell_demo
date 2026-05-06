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
    error: str
    last_failed_step: str
    retry_count: int
    max_retries: int


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


def _execute_step_with_retry(state: AgentState, step_name: str, step_fn) -> AgentState:
    try:
        updated_state = step_fn(state)
        return {
            **updated_state,
            "error": "",
            "last_failed_step": "",
            "retry_count": 0,
        }
    except Exception as exc:
        return {
            **state,
            "error": str(exc),
            "last_failed_step": step_name,
            "retry_count": state.get("retry_count", 0) + 1,
        }


def _retry_decision(state: AgentState) -> str:
    if not state.get("error"):
        return "SUCCESS"
    if state.get("retry_count", 0) <= state.get("max_retries", 2):
        return "RETRY"
    return "FAIL"


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
Column-mapping rules:
- User wording may be business-friendly, abbreviated, or non-exact; map it to the closest semantically correct table/column names from Schema Context.
- Use ONLY table/column names that exist in Schema Context (do not invent names).
- If multiple candidate columns are similar, choose the most context-appropriate one based on table purpose and user intent.
- Prefer exact schema column names in final SQL even when user does not provide exact names.
Domain guidance for STDBCOD:
- Source table for COD order issue tracking is `edm_cod_jsm_dly` (new COD order is loaded via the `issue_type` column).
- First validate table and column availability from Schema Context before choosing date columns.
- For creation logic in `edm_cod_jsm_dly`, use `edm_cod_jsm_dly.dice_ins_dt` as primary; if unavailable, fall back to `edm_cod_jsm_dly.dice_ins_crt_dt`.
- For creation logic in other tables, use that table's creation audit column with priority: `dice_ins_crt_dt` first, then `dice_ins_dt` if `dice_ins_crt_dt` is unavailable.
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


def fail_with_error(state: AgentState) -> AgentState:
    step = state.get("last_failed_step") or "unknown_step"
    retries = state.get("retry_count", 0)
    error = state.get("error", "Unknown error")
    return {
        **state,
        "answer": f"Request failed at '{step}' after {retries} retries. Error: {error}",
    }


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("decide_route", lambda state: _execute_step_with_retry(state, "decide_route", decide_route))
    graph.add_node("general_answer", lambda state: _execute_step_with_retry(state, "general_answer", general_answer))
    graph.add_node("retrieve_schema", lambda state: _execute_step_with_retry(state, "retrieve_schema", retrieve_schema))
    graph.add_node("generate_sql", lambda state: _execute_step_with_retry(state, "generate_sql", generate_sql))
    graph.add_node("execute_sql", lambda state: _execute_step_with_retry(state, "execute_sql", execute_sql))
    graph.add_node("draft_answer", lambda state: _execute_step_with_retry(state, "draft_answer", draft_answer))
    graph.add_node("fail_with_error", fail_with_error)

    graph.set_entry_point("decide_route")
    graph.add_conditional_edges(
        "decide_route",
        lambda state: state["route"] if _retry_decision(state) == "SUCCESS" else _retry_decision(state),
        {
            "DB_QUERY": "retrieve_schema",
            "GENERAL_CHAT": "general_answer",
            "RETRY": "decide_route",
            "FAIL": "fail_with_error",
        },
    )
    graph.add_conditional_edges(
        "general_answer",
        _retry_decision,
        {"SUCCESS": END, "RETRY": "general_answer", "FAIL": "fail_with_error"},
    )
    graph.add_conditional_edges(
        "retrieve_schema",
        _retry_decision,
        {"SUCCESS": "generate_sql", "RETRY": "retrieve_schema", "FAIL": "fail_with_error"},
    )
    graph.add_conditional_edges(
        "generate_sql",
        _retry_decision,
        {"SUCCESS": "execute_sql", "RETRY": "generate_sql", "FAIL": "fail_with_error"},
    )
    graph.add_conditional_edges(
        "execute_sql",
        _retry_decision,
        {"SUCCESS": "draft_answer", "RETRY": "execute_sql", "FAIL": "fail_with_error"},
    )
    graph.add_conditional_edges(
        "draft_answer",
        _retry_decision,
        {"SUCCESS": END, "RETRY": "draft_answer", "FAIL": "fail_with_error"},
    )
    graph.add_edge("fail_with_error", END)

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
        "error": "",
        "last_failed_step": "",
        "retry_count": 0,
        "max_retries": 2,
    }
    return app.invoke(initial_state)
