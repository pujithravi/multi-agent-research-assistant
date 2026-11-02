# Multi-Agent Research Assistant

Four cooperating agents research any topic end-to-end:

```
SearchAgent → SummarizeAgent → FactCheckAgent → WriteAgent
```

- **SearchAgent** — queries the web via Tavily and collects raw sources
- **SummarizeAgent** — condenses sources into key findings with citations
- **FactCheckAgent** — cross-checks the summary against sources, flags unverified/conflicting claims
- **WriteAgent** — writes the final structured report (intro, findings, confidence/caveats, sources)

Built with **LangGraph** (agent orchestration), **Google Gemini** (reasoning), **Tavily** (web search), and a **Streamlit** UI.

## 1. Get API keys (both free)

- Gemini: https://aistudio.google.com/apikey
- Tavily: https://tavily.com (free tier: 1,000 searches/month)

## 2. Setup

```bash
cd multi-agent-research-assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and paste in your two API keys.

## 3. Run

```bash
streamlit run app.py
```

This opens a browser tab at `http://localhost:8501`. Type a research question and watch the four agents hand off work in real time, ending in a cited report.

## 4. Run without the UI (optional, for testing/scripting)

```python
from dotenv import load_dotenv
load_dotenv()
from agents.graph import run_research

result = run_research("Impact of small modular nuclear reactors on grid decarbonization")
print(result["final_report"])
```

## Project structure

```
multi-agent-research-assistant/
├── agents/
│   └── graph.py       # LangGraph state machine + all 4 agent nodes
├── app.py              # Streamlit UI
├── requirements.txt
├── .env.example
└── README.md
```
