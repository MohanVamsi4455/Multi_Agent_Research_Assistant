import re
import time

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import ToolMessage

from src.agents.agents import search_agent, reader_agent, writer_chain, critic_chain
from src.tools.tools import scrape_url

URL_RE = re.compile(r'https?://[^\s\)\]\>"\'`]+')
SCORE_RE = re.compile(r'Score:\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10', re.IGNORECASE)

# LangGraph's own default is 10007 steps — effectively unbounded. An agent that
# keeps re-calling its tool would burn tokens until the API complained, so cap it.
# Each tool call costs ~2 steps (model turn + tool turn), so 12 allows ~5 calls.
AGENT_RECURSION_LIMIT = 12

# How many extra sources to fetch after the Reader agent picks its one.
# A report resting on a single source cannot score well, however it is written.
EXTRA_SOURCES = 2
MIN_USABLE_CHARS = 400

# scrape_url returns these as ordinary strings, so a length check alone is not
# enough — an error message is long enough to look like content.
SCRAPE_FAILURES = (
    "Could not scrape URL", "Could not extract", "HTTP error occurred",
    "Request timed out", "Connection dropped",
)

# Index/listing pages scrape to navigation text, not article prose.
INDEX_PATHS = {"", "/", "/blog", "/news", "/articles", "/posts", "/research", "/resources"}


def usable_scrape(text: str) -> bool:
    """Is this real article content, or an error string / nav menu?"""
    body = (text or "").strip()
    return len(body) >= MIN_USABLE_CHARS and not body.startswith(SCRAPE_FAILURES)


def looks_like_article(url: str) -> bool:
    from urllib.parse import urlparse
    path = urlparse(url).path.rstrip("/")
    return path.lower() not in INDEX_PATHS


def gather_extra_sources(urls, already: set, limit: int = EXTRA_SOURCES) -> list:
    """Scrape additional sources directly — no LLM in this loop.

    Deterministic on purpose: URL choice by code, validated by code. The Reader
    agent still does the judgement call on the primary source.
    """
    gathered = []
    for url in urls:
        if len(gathered) >= limit:
            break
        if url in already or not looks_like_article(url):
            continue
        already.add(url)
        body = scrape_url.invoke({"url": url})
        if usable_scrape(body):
            gathered.append({"url": url, "content": body})
    return gathered

STAGES = [
    ("search", "Search Agent", "Querying the web with Tavily"),
    ("read", "Reader Agent", "Scraping the most relevant source"),
    ("write", "Writer Chain", "Drafting the research report"),
    ("critic", "Critic Chain", "Scoring and reviewing the draft"),
]


class StageUsage(UsageMetadataCallbackHandler):
    """Token accounting for one pipeline stage.

    An agent makes several model calls inside a single `invoke` (one per loop
    turn); this aggregates all of them, so the number is what the stage really
    cost, not just its final call.
    """

    def __init__(self):
        super().__init__()
        self.llm_calls = 0

    def on_llm_end(self, response, **kwargs):
        self.llm_calls += 1
        return super().on_llm_end(response, **kwargs)

    def totals(self) -> dict:
        totals = {"input": 0, "output": 0, "total": 0, "cached": 0,
                  "calls": self.llm_calls, "models": sorted(self.usage_metadata)}
        for usage in self.usage_metadata.values():
            totals["input"] += usage.get("input_tokens", 0) or 0
            totals["output"] += usage.get("output_tokens", 0) or 0
            totals["total"] += usage.get("total_tokens", 0) or 0
            totals["cached"] += (usage.get("input_token_details") or {}).get("cache_read", 0) or 0
        return totals


def _agent_config(usage: StageUsage) -> dict:
    return {"callbacks": [usage], "recursion_limit": AGENT_RECURSION_LIMIT}


def as_text(content) -> str:
    """Flatten a message's content to plain text.

    Gemini returns content as a list of blocks, e.g.
    [{'type': 'text', 'text': '...', 'extras': {'signature': '...'}}].
    Rendering that raw leaks the thought-signature blob into prompts and UI,
    so everything downstream goes through here.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def tool_output(result) -> str:
    """The untouched tool results from an agent run, before the LLM rewrote them."""
    return "\n\n".join(
        as_text(m.content)
        for m in result.get("messages", [])
        if isinstance(m, ToolMessage)
    )


def tool_call_count(result) -> int:
    return sum(1 for m in result.get("messages", []) if isinstance(m, ToolMessage))


def extract_urls(*texts) -> list:
    seen, urls = set(), []
    for text in texts:
        for url in URL_RE.findall(text or ""):
            url = url.rstrip('.,;:')
            if url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def extract_score(feedback: str):
    match = SCORE_RE.search(feedback or "")
    return float(match.group(1)) if match else None


def run_research_pipeline_stream(topic: str):
    """Run the pipeline, yielding an event per stage so a UI can follow along.

    Events: {"type": "start"|"done", "stage": str, ...} then {"type": "finished", "state": dict}.
    Each "done" event carries that stage's token usage.
    """
    state = {"topic": topic, "usage": {}, "tool_calls": {}}
    started = time.time()

    # ── Step 1 — Search agent ──────────────────────────────
    yield {"type": "start", "stage": "search"}

    usage = StageUsage()
    search_result = search_agent().invoke(
        {"messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]},
        config=_agent_config(usage),
    )
    state["search_results"] = as_text(search_result["messages"][-1].content)
    state["search_raw"] = tool_output(search_result)
    state["usage"]["search"] = usage.totals()
    state["tool_calls"]["search"] = tool_call_count(search_result)

    yield {"type": "done", "stage": "search", "content": state["search_results"],
           "usage": state["usage"]["search"]}

    # ── Step 2 — Reader agent ──────────────────────────────
    yield {"type": "start", "stage": "read"}

    usage = StageUsage()
    reader_result = reader_agent().invoke(
        {"messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content. "
            f"Prefer a specific article over a homepage or index page.\n\n"
            f"Search Results:\n{state['search_raw'] or state['search_results']}"
        )]},
        config=_agent_config(usage),
    )
    state["scraped_content"] = as_text(reader_result["messages"][-1].content)
    state["scraped_raw"] = tool_output(reader_result)
    state["usage"]["read"] = usage.totals()
    state["tool_calls"]["read"] = tool_call_count(reader_result)

    # Broaden the evidence base before writing. The Reader read one source well;
    # these add breadth so the report is not resting on a single page.
    primary = set(extract_urls(state["scraped_content"]))
    state["extra_sources"] = gather_extra_sources(
        extract_urls(state["search_raw"], state["search_results"]), primary
    )

    yield {"type": "done", "stage": "read", "content": state["scraped_content"],
           "usage": state["usage"]["read"]}

    # ── Step 3 — Writer chain ──────────────────────────────
    yield {"type": "start", "stage": "write"}

    extra_blocks = "\n\n".join(
        f"--- ADDITIONAL SOURCE: {s['url']} ---\n{s['content']}"
        for s in state["extra_sources"]
    )
    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
        + (f"\n\n{extra_blocks}" if extra_blocks else "")
    )
    state["research"] = research_combined
    usage = StageUsage()
    state["report"] = writer_chain.invoke(
        {"topic": topic, "research": research_combined},
        config={"callbacks": [usage]},
    )
    state["usage"]["write"] = usage.totals()
    state["tool_calls"]["write"] = 0

    yield {"type": "done", "stage": "write", "content": state["report"],
           "usage": state["usage"]["write"]}

    # ── Step 4 — Critic chain ──────────────────────────────
    yield {"type": "start", "stage": "critic"}

    usage = StageUsage()
    # The critic now sees the research too. Grading citations without access to
    # the sources forced it to treat every claim as unverifiable.
    state["feedback"] = critic_chain.invoke(
        {"report": state["report"], "research": research_combined[:18000]},
        config={"callbacks": [usage]},
    )
    state["usage"]["critic"] = usage.totals()
    state["tool_calls"]["critic"] = 0

    yield {"type": "done", "stage": "critic", "content": state["feedback"],
           "usage": state["usage"]["critic"]}

    state["sources"] = extract_urls(state["search_raw"], state["search_results"], state["report"])
    state["score"] = extract_score(state["feedback"])
    state["duration"] = round(time.time() - started, 1)
    state["usage_total"] = {
        key: sum(stage[key] for stage in state["usage"].values())
        for key in ("input", "output", "total", "cached", "calls")
    }

    yield {"type": "finished", "state": state}


def run_research_pipeline(topic: str) -> dict:
    """Console entry point — same pipeline, printed instead of streamed."""
    labels = {key: (label, desc) for key, label, desc in STAGES}
    state = {}

    for event in run_research_pipeline_stream(topic):
        if event["type"] == "start":
            label, desc = labels[event["stage"]]
            print("\n" + "=" * 60)
            print(f"{label} — {desc} ...")
            print("=" * 60)
        elif event["type"] == "done":
            use = event["usage"]
            print(f"\n{event['content']}")
            print(f"\n[tokens] in {use['input']:,} · out {use['output']:,} · "
                  f"total {use['total']:,} · {use['calls']} model call(s)")
        elif event["type"] == "finished":
            state = event["state"]

    if state:
        total = state["usage_total"]
        print("\n" + "=" * 60)
        print(f"TOTAL — {total['total']:,} tokens "
              f"(in {total['input']:,} / out {total['output']:,}) "
              f"across {total['calls']} model calls in {state['duration']}s")
        print("=" * 60)

    return state
