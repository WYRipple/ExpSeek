import json
import os
import datetime
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Dict, Optional

from expseek.tools.tool_manager import BaseTool, register_tool
from expseek.agent.prompt import EXTRACTOR_PROMPT
from expseek.llm.client import get_llm_summary


# Maximum webpage content length to process
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))


class WebMemoryCache:
    """
    In-memory webpage cache backed by a JSONL file.
    Avoids redundant Jina API calls for previously visited URLs.
    """

    def __init__(self, cache_file_path: str, lock):
        self.cache_file_path = cache_file_path
        self.cache: Dict[str, str] = {}
        self.lock = lock
        self._load_cache()

    def _load_cache(self):
        """Load cached pages from JSONL file into memory."""
        if not os.path.exists(self.cache_file_path):
            os.makedirs(os.path.dirname(self.cache_file_path), exist_ok=True)
            open(self.cache_file_path, 'a').close()
            return

        try:
            with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            if 'url' in data and 'content' in data:
                                self.cache[data['url']] = data['content']
                        except json.JSONDecodeError:
                            continue
            print(f"[WebMemoryCache] Loaded {len(self.cache)} cached pages.")
        except Exception as e:
            print(f"[WebMemoryCache] Error loading cache: {e}")

    def get(self, url: str) -> Optional[str]:
        """Retrieve cached content for a URL."""
        with self.lock:
            return self.cache.get(url)

    def save(self, url: str, content: str):
        """Save webpage content to memory cache and append to JSONL file."""
        with self.lock:
            self.cache[url] = content
            try:
                with open(self.cache_file_path, 'a', encoding='utf-8') as f:
                    data = {
                        "url": url,
                        "content": content,
                        "timestamp": str(datetime.datetime.now())
                    }
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
                print(f"[WebMemoryCache] Cached URL: {url[:50]}...")
            except Exception as e:
                print(f"[WebMemoryCache] Error saving to file: {e}")


@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    """
    Visit tool that reads webpage content via Jina Reader API,
    then uses an LLM to extract goal-relevant information.
    """

    name = 'visit'
    description = (
        'Visit a webpage and return the answer and the raw content snippet '
        'corresponding to the answer based on the visit goal. '
        'Before using this tool, you must think and determine the visit goal. '
        'A website may have multiple visit goals, but they should be within one string.'
    )
    parameters = {
        "type": "object",
        "arguments": {
            "url": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "minItems": 1,
                "description": "The URL(s) of the website(s) to visit; can be one or a list of URLs."
            },
            "goal": {
                "type": "string",
                "description": "The visit goal, i.e., what information you want to obtain from these sites."
            }
        },
        "required": ["url", "goal"]
    }

    def __init__(self, config=None, root_dir=None, lock=None, **kwargs):
        super().__init__()
        self.config = config
        self.root_dir = root_dir
        self.my_lock = lock
        self.web_cache = WebMemoryCache(config.visit_path, self.my_lock)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """Execute visit for one or multiple URLs."""
        try:
            url = params["url"]
            goal = params["goal"]
        except Exception:
            return "[Visit] Invalid request format: input must be a JSON object with 'url' and 'goal' fields."

        if isinstance(url, str):
            response, _ = self.readpage(url, goal)
        else:
            assert isinstance(url, List)
            responses = []
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(self.readpage, u, goal): u for u in url}
                for future in as_completed(futures):
                    try:
                        result, _ = future.result()
                        responses.append(result)
                    except Exception as e:
                        responses.append(f"[Visit] Error fetching {futures[future]}: {str(e)}")
            response = "\n=======\n".join(responses)

        return response.strip()

    def call_server(self, msgs) -> str:
        """Call the summary LLM and ensure the response is valid JSON."""
        content = get_llm_summary(msgs, self.config)
        if content:
            try:
                json.loads(content)
            except Exception:
                # Try to extract JSON substring
                left = content.find('{')
                right = content.rfind('}')
                if left != -1 and right != -1 and left <= right:
                    content = content[left:right + 1]
            return content
        return "None"

    def jina_readpage(self, url: str) -> str:
        """
        Fetch raw webpage content via Jina Reader API.
        Returns cached content if available.
        """
        cached_content = self.web_cache.get(url)
        if cached_content:
            print(f"[Visit] Cache hit for URL: {url[:50]}...")
            return cached_content

        headers = {"Authorization": f"Bearer {self.config.jina_key}"}
        max_retries = 3
        timeout = 30

        for attempt in range(max_retries):
            try:
                print("[Visit] Calling Jina API...")
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=timeout
                )
                print("[Visit] Jina response received.")
                if response.status_code == 200:
                    webpage_content = response.text
                    if webpage_content and not webpage_content.startswith("QUIT:"):
                        self.web_cache.save(url, webpage_content)
                    return webpage_content
                else:
                    raise ValueError(f"Jina returned HTTP {response.status_code}")
            except Exception as e:
                if attempt == max_retries - 1:
                    return "QUIT: [Visit] Failed to read page."
                print(f"[Visit] Retry {attempt + 1}/{max_retries}: {str(e)}")

    def readpage(self, url: str, goal: str) -> tuple:
        """
        Read a webpage and extract goal-relevant information using the summary LLM.
        Returns (useful_information, raw_content).
        """
        failed_response = (
            f"The useful information in {url} for user goal {goal} as follows:\n\n"
            f"Evidence in page:\nThe provided webpage content could not be accessed. "
            f"Please check the URL or file format.\n\n"
            f"Summary:\nThe webpage content could not be processed, "
            f"and therefore, no information is available.\n\n"
        )

        max_attempts = 10
        for attempt in range(max_attempts):
            raw_content = self.jina_readpage(url)

            valid = (
                raw_content
                and not raw_content.startswith("[visit][jina_readpage] Failed to read page.")
                and raw_content != "[visit][jina_readpage] Empty content."
                and not raw_content.startswith("[document_parser]")
            )

            if valid:
                raw_content = raw_content[:WEBCONTENT_MAXLENGTH]
                messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(
                    webpage_content=raw_content, goal=goal
                )}]
                raw = self.call_server(messages)

                # If response is too short, the page may be too long; truncate and retry
                summary_retries = 3
                while len(raw) < 10 and summary_retries >= 0:
                    truncate_length = int(0.7 * len(raw_content)) if summary_retries > 0 else 25000
                    print(
                        f"[Visit] Summary too short for {url}, "
                        f"truncating to {truncate_length} chars "
                        f"(retries left: {summary_retries})"
                    )
                    raw_content = raw_content[:truncate_length]
                    messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(
                        webpage_content=raw_content, goal=goal
                    )}]
                    raw = self.call_server(messages)
                    summary_retries -= 1

                # Try to parse JSON response
                parse_retry_times = 0
                while parse_retry_times < 3:
                    try:
                        raw = json.loads(raw)
                        break
                    except Exception:
                        raw = self.call_server(messages)
                        parse_retry_times += 1

                # Build output based on parse result
                if parse_retry_times >= 3:
                    return failed_response, raw_content
                else:
                    useful_information = (
                        f"The useful information in {url} for user goal {goal} as follows:\n\n"
                        f"Evidence in page:\n{str(raw['evidence'])}\n\n"
                        f"Summary:\n{str(raw['summary'])}\n\n"
                    )
                    return useful_information, raw_content

            if attempt == max_attempts - 1:
                return failed_response, raw_content
