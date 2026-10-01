import requests
import os
import time
from dotenv import load_dotenv
from langchain.tools import tool
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tavily import TavilyClient
from rich import print
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
import re


load_dotenv()

tavily_api_key=os.getenv("TAVILY_API_KEY")

tavily_client = TavilyClient(api_key=tavily_api_key)


# Dropped connections are routine against both the search API and arbitrary
# websites. RemoteDisconnected surfaces as a plain requests.ConnectionError —
# Tavily's own error classes do not cover it — so handle it here.
TRANSIENT_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _with_retry(call, attempts: int = 3, base_delay: float = 1.0):
    """Run `call`, retrying transient network failures with exponential backoff."""
    last = None
    for attempt in range(attempts):
        try:
            return call()
        except TRANSIENT_ERRORS as exc:
            last = exc
            if attempt < attempts - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise last


def _session() -> requests.Session:
    """A session that retries dropped connections and throttling responses."""
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


@tool
def web_search(query:str)->str :
    """ Searches the web for the given query and returns the results. """
    out=[]

    try:
        results = _with_retry(lambda: tavily_client.search(query=query, num_results=3))
    except TRANSIENT_ERRORS as exc:
        return (
            f"SEARCH FAILED — the connection to the search API dropped after 3 attempts "
            f"({type(exc).__name__}). This is usually transient. Try one more search; if it "
            f"fails again, report that web search is unavailable and do not invent results."
        )
    except Exception as exc:
        return (
            f"SEARCH FAILED — {type(exc).__name__}: {exc}. "
            f"Do not invent results; report that the search could not be completed."
        )

    for r in results.get('results', []):
        out.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n")

    if not out:
        return "SEARCH RETURNED NO RESULTS. Try a different, broader query."

    return "\n----\n".join(out)


@tool
def scrape_url(url: str) -> str:
    """
    Scrape and extract clean readable content from a URL.
    Uses multiple extraction strategies for better reliability.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    try:
        # ── Fetch page ─────────────────────────────────────
        # Session retries dropped connections and 429/5xx before giving up.
        response = _session().get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        html = response.text

        # ──────────────────────────────────────────────────
        # Strategy 1 → trafilatura (BEST for articles/blogs)
        # ──────────────────────────────────────────────────
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False
        )

        if extracted and len(extracted.strip()) > 200:
            cleaned = re.sub(r'\s+', ' ', extracted)
            return cleaned[:5000]

        # ──────────────────────────────────────────────────
        # Strategy 2 → readability
        # ──────────────────────────────────────────────────
        doc = Document(html)
        clean_html = doc.summary()

        soup = BeautifulSoup(clean_html, "html.parser")

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        if text and len(text.strip()) > 200:
            cleaned = re.sub(r'\s+', ' ', text)
            return cleaned[:5000]

        # ──────────────────────────────────────────────────
        # Strategy 3 → fallback full page extraction
        # ──────────────────────────────────────────────────
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        cleaned = re.sub(r'\s+', ' ', text)

        if cleaned:
            return cleaned[:5000]

        return "Could not extract meaningful content from the page."

    except requests.exceptions.Timeout:
        return "Request timed out while scraping the URL."

    except requests.exceptions.ConnectionError as e:
        return f"Connection dropped while scraping the URL (retried 3 times): {str(e)}"

    except requests.exceptions.HTTPError as e:
        return f"HTTP error occurred: {str(e)}"

    except Exception as e:
        return f"Could not scrape URL: {str(e)}"
    