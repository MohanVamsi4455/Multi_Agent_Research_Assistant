# Multi-Agent Research Assistant

A research pipeline that turns a single topic into a sourced, structured report — then grades its own work.

Four components run in sequence: a **search agent** queries the live web, a **reader agent** picks the most promising result and reads it in full, a **writer chain** drafts a cited report, and a **critic chain** scores it out of 10 against the evidence it was given. The whole run is driven from a Streamlit UI that animates each stage as it executes and accounts for every token it spends.

Built with LangChain `create_agent`, Google Gemini, and Tavily.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.4-1C3C3C)](https://python.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-6366F1.svg)](LICENSE)

> **Screenshot:** add one at `docs/screenshot.png` and reference it here — the live pipeline tracker is the thing worth showing.

---

## Architecture

```
                   ┌────────────────────────────────────────┐
                   │   app.py (UI)      main.py (CLI)       │
                   └───────────────────┬────────────────────┘
                                       │  topic
                                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  pipeline.py — run_research_pipeline_stream()                    │
│  four stages in order, one event yielded after each              │
└──────┬─────────────────┬─────────────────┬─────────────────┬─────┘
       ▼                 ▼                 ▼                 ▼
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│  1 Search  │    │  2 Reader  │    │  3 Writer  │    │  4 Critic  │
│   Agent    │───►│   Agent    │───►│   Chain    │───►│   Chain    │
│ tool loop  │    │ tool loop  │    │   1 pass   │    │   1 pass   │
└──────┬─────┘    └──────┬─────┘    └────────────┘    └──────┬─────┘
       │                 │                                   │
       ▼                 ▼                                   ▼
┌────────────┐    ┌────────────┐                      ┌────────────┐
│ web_search │    │ scrape_url │                      │Score: X/10 │
│ Tavily API │    │3 extractors│                      │  + review  │
└────────────┘    └────────────┘                      └────────────┘
```

Data flows left to right: the search agent's findings and its raw tool output go to the reader agent, the reader's extracted article goes to the writer, and the writer's report goes to the critic along with the research it was written from. The pipeline also scrapes two extra sources itself, between stages 2 and 3, with no model in the loop.

**Agents vs. chains.** Steps 1 and 2 are true agents: they hold a tool, decide when to call it, and loop until satisfied. Steps 3 and 4 need no tools — they are single-pass LCEL chains. Using an agent where a chain suffices buys you latency and nondeterminism for nothing.

**What a run returns.** One dict, carried by the final `finished` event: the report, the critic's review, the parsed score, the deduplicated source URLs, per-stage token usage, and wall-clock duration — plus the raw tool output from every stage, which is what the Agent Trace tab reads.

---

## How it works

| Stage | Type | Tool | Output |
|---|---|---|---|
| **Search Agent** | `create_agent` | `web_search` (Tavily) | Findings brief, notable sources, gaps |
| **Reader Agent** | `create_agent` | `scrape_url` | Up to 5,000 chars of article text, detail preserved |
| **Breadth pass** | plain Python | `scrape_url` | 2 extra validated sources, no model call |
| **Writer Chain** | LCEL | — | Report: intro, ≥3 findings, analysis, limitations, sources |
| **Critic Chain** | LCEL | — | `Score: X/10`, strengths, fixes, unsupported claims, gaps |

### Extraction

`scrape_url` tries three strategies in order and takes the first that yields more than 200 characters:

1. **trafilatura** — best for articles and blog posts
2. **readability-lxml** + BeautifulSoup — fallback for awkward markup
3. **Raw BeautifulSoup** — strips `script`/`style`/`nav`/`footer`/`header`/`aside`/`form` and takes what's left

Both tools return plain strings, including on failure, so the pipeline cannot tell success from failure by length alone. `usable_scrape()` closes that hole: at least 400 characters **and** not starting with a known failure prefix. `looks_like_article()` rejects bare domains and section roots (`/blog`, `/news`, …) before a request is even made — index pages scrape to navigation text, which reads like content and is worth nothing.

### Resilience

- Tavily calls retry transient network errors three times with exponential backoff (`_with_retry`).
- Scrapes use a `requests.Session` with `urllib3` `Retry` on connect/read failures and on 429/500/502/503/504.
- Agents run under `recursion_limit = 12` (~5 tool calls). LangGraph's default of 10007 steps means a looping agent would burn tokens until the provider complained.
- The UI maps raw exceptions to an actionable message (`describe_failure`) — network, quota, or credentials — instead of a traceback.

---

## Quickstart

**Prerequisites:** Python 3.11+, a [Google AI Studio](https://aistudio.google.com/app/apikey) key, and a [Tavily](https://tavily.com/) key (free tier is enough).

```bash
git clone https://github.com/MohanVamsi4455/Multi_Agent_Research_Assistant.git
cd Multi_Agent_Research_Assistant

python -m venv multi_agent_venv
multi_agent_venv\Scripts\activate        # Windows
# source multi_agent_venv/bin/activate   # macOS / Linux

pip install -r requirement.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_google_ai_studio_key
TAVILY_API_KEY=your_tavily_key
```

Run the UI:

```bash
streamlit run app.py
```

Or run headless from the terminal:

```bash
python main.py "agentic AI in production"
```

A typical run takes **30–60 seconds** end to end. Verified against `langchain 1.4.2`, `langchain-google-genai 4.4.0`, `streamlit 1.64.0`, `tavily-python 0.8.4` and `trafilatura 2.2.0` on Python 3.11.

---

## The UI

- **Live flow graph** — an SVG of the real call graph: each tool sits above the agent that owns it, with `call` / `result` arrows between them, and data flows left to right along the spine. The active node glows and its edges animate in the direction of travel; completed nodes turn green. Hover any node for detail.
- **Metrics strip** — critic score ring, sources found, report word count, wall-clock duration.
- **Five tabs** — Report · Critic Review · Sources · Token Usage · Agent Trace.
- **Downloads** — report only, or the full run including review and sources.

### The Token Usage tab

`StageUsage` subclasses `UsageMetadataCallbackHandler` and counts model calls as well as tokens, so a stage that looped shows what the loop cost:

```
Search Agent    3 model calls    18,402 tokens
Reader Agent    2 model calls    11,118 tokens
```

An agent's `invoke()` makes one model call per tool-loop turn. Reading only the final response undercounts a multi-turn agent badly.

### The Agent Trace tab

A LangChain agent run ends with an `AIMessage` that is the model's *summary of* the tool result, not the tool result itself — the raw `ToolMessage` sits earlier in the message list and is usually discarded. The trace tab displays both, with character counts, so the compression is visible:

```
Reader — raw scrape      5,000 chars
Reader — agent summary   1,635 chars
```

Knowing where that loss happens is most of what separates a pipeline that works from one that merely runs.

---

## Project structure

```
├── app.py                      # Streamlit UI — styles, flow graph, tabs, error copy
├── main.py                     # CLI entry point (UTF-8 safe console)
├── requirement.txt
├── .streamlit/config.toml      # Theme
└── src/
    ├── agents/agents.py        # LLMs, system prompts, 2 agents, 2 LCEL chains
    ├── pipeline/pipeline.py    # Orchestration, breadth pass, token accounting
    └── tools/tools.py          # web_search, scrape_url, retry helpers
```

`run_research_pipeline_stream()` is a generator that yields a `start` / `done` event per stage, which is what lets the UI update live. `run_research_pipeline()` wraps the same generator and prints instead — one implementation, two front ends.

---

## Configuration

| What | Where | Default |
|---|---|---|
| Model | `src/agents/agents.py` | `gemini-3.1-flash-lite` |
| Temperature | `src/agents/agents.py` | `0` (writer: `0.3`) |
| Agent step cap | `src/pipeline/pipeline.py` | `AGENT_RECURSION_LIMIT = 12` |
| Extra sources | `src/pipeline/pipeline.py` | `EXTRA_SOURCES = 2` |
| Usable-scrape floor | `src/pipeline/pipeline.py` | `MIN_USABLE_CHARS = 400` |
| Research sent to critic | `src/pipeline/pipeline.py` | first 18,000 chars |
| Scrape cap | `src/tools/tools.py` | 5,000 chars |
| Request timeout | `src/tools/tools.py` | 15s |
| Theme | `.streamlit/config.toml` | Dark, indigo accent |

---

## Engineering notes

Things worth knowing before you build on this:

- **Tool results reach the model as plain text.** Both tools return `str`, so there is no schema enforcement on what the LLM sees. `create_agent` accepts a `response_format` if you want typed output back — a natural next step.
- **Gemini returns block-structured content.** `message.content` is a `list[dict]`, not a `str`, and each block carries a base64 thought-`signature`. `as_text()` in `pipeline.py` flattens it and strips the blob — without that, the signature leaks into downstream prompts and burns tokens.
- **The reader sees raw search output.** It receives the unmodified `Title:/URL:/Snippet:` rows rather than the search agent's prose, so it selects from real URLs. Feeding it the summary instead tends to land it on homepages and index pages.
- **The critic sees the research, not just the report.** Grading citations without access to the sources forced it to treat every claim as unverifiable, which flattened every score. With both in hand it can separate a writing failure from a research gap — and it reports them under different headings.
- **The score's first line is a contract.** `extract_score()` parses `Score: X/10` out of the critic's response and the UI renders it as a ring. Change that prompt's output format and the metric silently goes blank.
- **Scores are not comparable across topics.** The critic grades a report against the research it was given, so a thin-evidence topic caps out lower by design. Use the score to compare drafts of one run, not two different runs.

---

## Roadmap

- [x] Read more than one source per run — a primary deep read plus a 2-source breadth pass
- [x] Per-stage token accounting surfaced in the UI
- [ ] `include_raw_content=True` on Tavily to skip scraping where full text is already returned
- [ ] Feed critic feedback back to the writer for a revision pass (currently the score is terminal)
- [ ] `response_format` on both agents for typed hand-offs
- [ ] Cache results per topic to make demos instant
- [ ] Scrape the breadth-pass sources concurrently — they are independent and currently serial

---

## License

[MIT](LICENSE) © MohanVamsi4455
