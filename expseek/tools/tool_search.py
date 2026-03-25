import json
import time
import requests
from typing import List, Union
from expseek.tools.tool_manager import BaseTool, register_tool
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore
from urllib.parse import urlencode
import langid

# Limit concurrent search requests
api_semaphore = Semaphore(20)


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    """
    Search tool powered by BrightData web scraping API.
    Returns top 5 search results (title, link, snippet) for each query.
    """

    name = "search"
    description = (
        "Execute a search: provide a list of query strings to search, "
        "the tool returns the top 5 website links and snippet summaries for each query. "
        "However, this tool can only provide direction for solving problems; "
        "if you wish to obtain accurate information, you must call the visit tool "
        "to enter a specific link for details."
    )
    parameters = {
        "type": "object",
        "arguments": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            }
        },
        "required": ["query"],
    }

    def __init__(self, config=None, root_dir=None, **kwargs):
        super().__init__()
        self.config = config
        self.root_dir = root_dir
        # BrightData API credentials, set in config
        self.api_key = config.brightdata_key
        self.zone = config.brightdata_zone
        self.location = config.brightdata_location

    def google_search(self, query: str) -> str:
        """
        Call BrightData API to fetch Google search results
        and return top 5 results in a structured format.
        """
        with api_semaphore:
            # Detect query language to set search locale
            lang_code, _ = langid.classify(query)
            if lang_code == "zh":
                mkt, setLang = "zh-CN", "zh"
            else:
                mkt, setLang = "en-US", "en"

            # Build target Google search URL
            encoded_query = urlencode({
                "q": query,
                "mkt": mkt,
                "setLang": setLang
            })
            target_url = f"https://www.google.com/search?{encoded_query}&brd_json=1&cc={self.location}"

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "zone": self.zone,
                "url": target_url,
                "format": "raw"
            }

            # Request with retry
            for i in range(5):
                try:
                    resp = requests.post(
                        "https://api.brightdata.com/request",
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
                    results = resp.json()
                    time.sleep(0.5)
                    break
                except Exception as e:
                    print(f"[Search] Attempt {i+1} failed: {e}")
                    time.sleep(2 ** i)
                    if i == 4:
                        return "Google search timeout, no results returned."

            if resp.status_code != 200:
                return f"[Search] Error: HTTP {resp.status_code} - {resp.text}"

            # Parse and format results
            try:
                if "organic" not in results:
                    return f"No results found for query '{query}'."

                web_snippets = []
                for idx, page in enumerate(results["organic"][:5], start=1):
                    title = page.get("title", "No title")
                    link = page.get("link", "")
                    date_published = "\nDate published: " + page["date"] if "date" in page else ""
                    source = "\nSource: " + page["source"] if "source" in page else ""
                    snippet = "\n" + page["description"] if "description" in page else ""
                    web_snippets.append(
                        f"{idx}. [{title}]({link}){date_published}{source}\n{snippet}"
                    )

                return (
                    f"A Google search for '{query}' found {len(web_snippets)} results:\n\n"
                    f"## Web Results\n" + "\n\n".join(web_snippets)
                )

            except Exception as e:
                return f"[Search] Failed to parse results for '{query}': {str(e)}"

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """Execute search for one or multiple queries."""
        assert self.api_key is not None, "[Search] BrightData API key is not set in config."

        try:
            query = params["query"]
        except Exception:
            return "[Search] Invalid request format: input must be a JSON object with 'query' field."

        if isinstance(query, str):
            return self.google_search(query)
        else:
            assert isinstance(query, List)
            with ThreadPoolExecutor(max_workers=3) as executor:
                resp_list = list(executor.map(self.google_search, query))
            return "\n=======\n".join(resp_list)
