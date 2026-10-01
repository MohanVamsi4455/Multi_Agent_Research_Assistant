"""CLI entry point. For the UI, run: streamlit run app.py"""

import sys

from src.pipeline.pipeline import run_research_pipeline

# Model output contains em dashes, smart quotes and arrows. The default Windows
# console codec (cp1252) cannot encode them and raises UnicodeEncodeError mid-print,
# so force UTF-8 and degrade gracefully if a glyph is still unmappable.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "New LLM Models"
    run_research_pipeline(topic)
