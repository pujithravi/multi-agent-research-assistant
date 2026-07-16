"""
Streamlit front-end for the Multi-Agent Research Assistant.

Run with:  streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

for _key in ("GOOGLE_API_KEY", "TAVILY_API_KEY"):
    if _key in st.secrets and not os.environ.get(_key):
        os.environ[_key] = st.secrets[_key]


from agents.graph import build_graph  # noqa: E402

st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="🔎", layout="wide")

st.title("🔎 Multi-Agent Research Assistant")
st.caption(
    "Search → Summarize → Fact-Check → Write, handled by four cooperating "
    "agents built with LangGraph + Gemini + Tavily."
)

missing_keys = [k for k in ("GOOGLE_API_KEY", "TAVILY_API_KEY") if not os.environ.get(k)]
if missing_keys:
    st.warning(
        f"Missing environment variable(s): {', '.join(missing_keys)}. "
        "Copy `.env.example` to `.env` and fill in your API keys before running a query."
    )

topic = st.text_input(
    "What do you want researched?",
    placeholder="e.g. Impact of small modular nuclear reactors on grid decarbonization",
)

run_clicked = st.button("Run research", type="primary", disabled=bool(missing_keys))

STAGE_LABELS = {
    "search": "🔍 SearchAgent — gathering sources",
    "summarize": "📝 SummarizeAgent — condensing findings",
    "fact_check": "✅ FactCheckAgent — verifying claims",
    "write": "🖋️ WriteAgent — writing final report",
}

if run_clicked and topic.strip():
    graph = build_graph()
    initial_state = {
        "topic": topic,
        "search_results": "",
        "summary": "",
        "fact_check_notes": "",
        "final_report": "",
        "log": [],
    }

    progress_box = st.status("Running agents...", expanded=True)
    final_state = None

    try:
        for step in graph.stream(initial_state):
            for node_name, node_state in step.items():
                label = STAGE_LABELS.get(node_name, node_name)
                progress_box.write(f"**{label}** — done")
                final_state = node_state

        progress_box.update(label="Research complete", state="complete", expanded=False)

        if final_state:
            st.subheader("Final Report")
            st.markdown(final_state["final_report"])

            with st.expander("Summary (SummarizeAgent output)"):
                st.markdown(final_state["summary"])

            with st.expander("Fact-check notes (FactCheckAgent output)"):
                st.markdown(final_state["fact_check_notes"])

            with st.expander("Raw sources (SearchAgent output)"):
                st.text(final_state["search_results"])

    except Exception as e:
        progress_box.update(label="Failed", state="error")
        st.error(f"Something went wrong: {e}")

elif run_clicked:
    st.error("Please enter a research topic.")
