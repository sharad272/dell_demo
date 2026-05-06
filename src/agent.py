import re
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


def _build_keyword_query(user_query: str) -> str:
    # Keep alphanumeric/underscore tokens and remove common stop words
    # so BM25-style lexical matching gets stronger signal.
    tokens = re.findall(r"[A-Za-z0-9_]+", user_query.lower())
    stop_words = {
        "the",
        "a",
        "an",
        "for",
        "to",
        "of",
        "in",
        "on",
        "with",
        "from",
        "show",
        "get",
        "find",
        "give",
        "me",
    }
    keywords = [t for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(dict.fromkeys(keywords))


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
    semantic_docs = vectorstore.similarity_search(state["user_query"], k=8)
    keyword_query = _build_keyword_query(state["user_query"])
    keyword_docs = vectorstore.similarity_search(keyword_query, k=8) if keyword_query else []

    # Hybrid merge: keep semantic ranking first, then append lexical matches.
    merged_docs = []
    seen = set()
    for doc in semantic_docs + keyword_docs:
        key = (
            doc.metadata.get("schema", ""),
            doc.metadata.get("table", ""),
            doc.metadata.get("column", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        merged_docs.append(doc)
        if len(merged_docs) >= 12:
            break

    context = "\n\n".join(d.page_content for d in merged_docs)
    return {**state, "schema_context": context}


def generate_sql(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_template(
        """You are an expert SQL Server assistant.
Given a user question and schema context, return ONLY one executable SQL Server query.
Instruction priority:
1) Explicit domain rules in this prompt.
2) Schema Context (actual available tables/columns).
3) User wording and intent.
4) General SQL best practices.
If any rule conflicts with a lower-priority instruction, always follow the higher-priority rule.

Hard constraints:
- Output SQL only. No prose, no markdown, no comments.
- Use ONLY tables/columns present in Schema Context. Never invent names.
- Map non-exact business words in user query to the closest schema column/table by meaning.
- Prefer simple, reliable SQL (avoid unnecessary CTEs/subqueries unless required).
- Use SQL Server syntax only.

Selection procedure (follow in order):
1) Identify the intent (detail lookup, count, trend, latest records, reporting summary, etc.).
2) Pick the primary table(s) from Schema Context that best match the intent.
3) Validate every selected column exists in those tables.
4) Build WHERE/GROUP BY/ORDER BY with intent-aligned columns and valid datatypes.

Domain rules for STDBCOD:
- COD issue tracking source table: `edm_cod_jsm_dly` (`issue_type` contains new COD order type info).
- Reporting/reference-style asks should prefer `jsm_cod_*_master` and `jsm_cod_*_mapping` tables.
- Creation time logic:
  - In `edm_cod_jsm_dly`: prefer `dice_ins_dt`; if unavailable, use `dice_ins_crt_dt`.
  - In other tables: prefer `dice_ins_crt_dt`; if unavailable, use `dice_ins_dt`.
- Updation time logic: use `dice_ins_upd_st` when available.
- Prefer `dice_` audit columns over non-audit date columns for created/updated/recency filters.
- Do not override these explicit domain rules based on generic keyword matching.

Quality checks before finalizing SQL:
- Ensure all referenced columns are present in Schema Context.
- Ensure join keys are valid columns from both joined tables.
- If user asks for latest/new/recent, apply ORDER BY on the correct creation/update audit column.
- If user asks for top N, use TOP (N).

Keep the Domain rules for STDBCOD as the highest priority.

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
