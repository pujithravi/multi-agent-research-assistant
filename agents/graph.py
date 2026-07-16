"""
Multi-Agent Research Assistant
-------------------------------
Four agents hand off work through a LangGraph state machine:

  SEARCH  ->  SUMMARIZE  ->  FACT-CHECK  ->  WRITE

- SearchAgent    : queries the web via Tavily and collects raw sources
- SummarizeAgent : condenses raw sources into key findings
- FactCheckAgent : cross-checks the summary against the sources, flags
                   unsupported or shaky claims
- WriteAgent     : produces the final polished report, citing sources

All reasoning steps use Google Gemini. Each node appends a human-readable
entry to `state["log"]` so the UI can show live agent progress.
"""

import os
from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient


# ---------------------------------------------------------------------------
# State shared between agents
# ---------------------------------------------------------------------------
class ResearchState(TypedDict):
    topic: str
    search_results: str
    summary: str
    fact_check_notes: str
    final_report: str
    log: List[str]


# ---------------------------------------------------------------------------
# Shared clients (lazily created so importing this module doesn't require
# API keys to be set until a graph is actually run)
# ---------------------------------------------------------------------------
def get_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to your .env file "
            "(see .env.example)."
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=temperature,
    )


def get_tavily() -> TavilyClient:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Add it to your .env file "
            "(see .env.example)."
        )
    return TavilyClient(api_key=api_key)


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------
def search_node(state: ResearchState) -> ResearchState:
    """SearchAgent: gathers raw information from the web via Tavily."""
    client = get_tavily()
    response = client.search(
        query=state["topic"],
        search_depth="advanced",
        max_results=6,
        include_answer=False,
    )

    formatted = []
    for i, result in enumerate(response.get("results", []), start=1):
        formatted.append(
            f"[Source {i}] {result.get('title', 'Untitled')}\n"
            f"URL: {result.get('url', '')}\n"
            f"Content: {result.get('content', '').strip()}\n"
        )

    search_results = "\n".join(formatted) if formatted else "No results found."

    return {
        **state,
        "search_results": search_results,
        "log": state["log"] + [f"SearchAgent: found {len(formatted)} sources for '{state['topic']}'."],
    }


def summarize_node(state: ResearchState) -> ResearchState:
    """SummarizeAgent: condenses the raw sources into key findings."""
    llm = get_llm(temperature=0.2)
    prompt = (
        "You are a research summarization agent. Read the sources below and "
        "produce a concise, well-organized summary of the key findings about "
        f"the topic: \"{state['topic']}\".\n\n"
        "Use short bullet points. After each point, note which source(s) "
        "support it, e.g. (Source 2).\n\n"
        f"SOURCES:\n{state['search_results']}"
    )
    response = llm.invoke(prompt)
    summary = response.content

    return {
        **state,
        "summary": summary,
        "log": state["log"] + ["SummarizeAgent: condensed sources into key findings."],
    }


def fact_check_node(state: ResearchState) -> ResearchState:
    """FactCheckAgent: cross-checks the summary against the original sources."""
    llm = get_llm(temperature=0.0)
    prompt = (
        "You are a fact-checking agent. Compare the SUMMARY against the "
        "SOURCES it was derived from. For each claim in the summary:\n"
        "- Confirm it if it is clearly supported by a source.\n"
        "- Flag it as 'UNVERIFIED' if no source directly supports it.\n"
        "- Flag it as 'CONFLICTING' if sources disagree.\n\n"
        "Be concise. Output a short list of notes, only flagging problems "
        "(if everything checks out, say so briefly).\n\n"
        f"SOURCES:\n{state['search_results']}\n\n"
        f"SUMMARY:\n{state['summary']}"
    )
    response = llm.invoke(prompt)
    fact_check_notes = response.content

    return {
        **state,
        "fact_check_notes": fact_check_notes,
        "log": state["log"] + ["FactCheckAgent: cross-checked summary against sources."],
    }


def write_node(state: ResearchState) -> ResearchState:
    """WriteAgent: produces the final polished, cited report."""
    llm = get_llm(temperature=0.4)
    prompt = (
        "You are a report-writing agent. Using the summary and fact-check "
        "notes below, write a clear, well-structured research report on:\n"
        f"\"{state['topic']}\"\n\n"
        "Structure it as:\n"
        "1. A short introduction\n"
        "2. Key findings (grouped logically, not just a bullet dump)\n"
        "3. A 'Confidence & Caveats' section based on the fact-check notes\n"
        "4. A Sources list (reuse source titles/URLs from the raw sources)\n\n"
        f"SUMMARY:\n{state['summary']}\n\n"
        f"FACT-CHECK NOTES:\n{state['fact_check_notes']}\n\n"
        f"RAW SOURCES (for citation):\n{state['search_results']}"
    )
    response = llm.invoke(prompt)
    final_report = response.content

    return {
        **state,
        "final_report": final_report,
        "log": state["log"] + ["WriteAgent: composed final report."],
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("search", search_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("fact_check", fact_check_node)
    graph.add_node("write", write_node)

    graph.add_edge(START, "search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", "fact_check")
    graph.add_edge("fact_check", "write")
    graph.add_edge("write", END)

    return graph.compile()


def run_research(topic: str) -> ResearchState:
    """Convenience entry point: run the full pipeline for a topic."""
    app = build_graph()
    initial_state: ResearchState = {
        "topic": topic,
        "search_results": "",
        "summary": "",
        "fact_check_notes": "",
        "final_report": "",
        "log": [],
    }
    return app.invoke(initial_state)
