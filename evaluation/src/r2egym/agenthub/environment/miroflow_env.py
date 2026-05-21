"""
MiroFlow-style Environment for web-research benchmarks (e.g. BrowseComp-EN).

Follows the same gym.Env interface as ``RepoEnv``
(reset  ->  add_commands  ->  step  ->  close)
but instead of executing bash commands in Docker, it dispatches tool calls
to lightweight HTTP-based tools (Serper search, free web scraper).

No extra dependencies beyond ``requests`` + ``beautifulsoup4``.
"""

import json
import os
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from r2egym.agenthub.action import Action
from r2egym.agenthub.observation import Observation
from r2egym.agenthub.utils.log import get_logger


# ---------------------------------------------------------------------------
# Tool implementations (pure HTTP, no MCP / external process)
# ---------------------------------------------------------------------------

MAX_SCRAPE_CHARS = 100_000  # truncate scraped pages to ~100 k chars


def _serper_search(
    q: str,
    gl: str = "us",
    hl: str = "en",
    num: int = 10,
    tbs: str = None,
    page: int = None,
    api_key: str = "",
    base_url: str = "https://google.serper.dev",
) -> str:
    """Call the Serper Google-search API and return JSON results."""
    if not api_key:
        return json.dumps({"error": "SERPER_API_KEY not set"})
    if not q or not q.strip():
        return json.dumps({"error": "Search query is empty"})

    payload: Dict[str, Any] = {"q": q.strip(), "gl": gl, "hl": hl, "num": num}
    if tbs:
        payload["tbs"] = tbs
    if page is not None:
        payload["page"] = page

    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    for attempt in range(3):
        try:
            resp = requests.post(
                f"{base_url}/search", json=payload, headers=headers, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            # Filter out HuggingFace dataset/space URLs
            organic = [
                r
                for r in data.get("organic", [])
                if not _is_hf_dataset_url(r.get("link", ""))
            ]
            # Re-try without quotes if zero results
            if not organic and '"' in q:
                payload["q"] = q.replace('"', "").strip()
                continue
            result = {
                "organic": organic,
                "searchParameters": data.get("searchParameters", {}),
            }
            return json.dumps(result, ensure_ascii=False)
        except requests.RequestException as e:
            if attempt == 2:
                return json.dumps({"error": f"Serper request failed: {e}"})
            time.sleep(2 ** attempt)

    return json.dumps({"error": "Serper search failed after retries"})


def _free_scrape(
    url: str,
    max_chars: int = MAX_SCRAPE_CHARS,
) -> str:
    """Scrape a URL using requests + BeautifulSoup (free, no API key)."""
    if not url or not url.strip():
        return json.dumps({"error": "URL is empty"})
    if _is_hf_dataset_url(url):
        return json.dumps(
            {"error": "Scraping HuggingFace datasets/spaces is not allowed."}
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    for attempt in range(3):
        try:
            resp = requests.get(
                url, headers=headers, timeout=30, allow_redirects=True
            )
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")

            # Non-HTML (PDF, plain text, etc.) – return raw text
            if "html" not in content_type:
                text = resp.text[:max_chars]
                return json.dumps(
                    {
                        "content": text,
                        "char_count": len(resp.text),
                        "truncated": len(resp.text) > max_chars,
                    },
                    ensure_ascii=False,
                )

            # HTML – extract readable text with BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove non-content elements
            for tag in soup(["script", "style", "nav", "footer", "header",
                             "noscript", "iframe", "svg", "form"]):
                tag.decompose()

            # Extract title
            title = soup.title.get_text(strip=True) if soup.title else ""

            # Extract text, preserving some structure
            lines = []
            if title:
                lines.append(f"# {title}\n")

            for el in soup.find_all(
                ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th",
                 "pre", "blockquote", "article", "section", "div"]
            ):
                text = el.get_text(separator=" ", strip=True)
                if not text:
                    continue
                tag_name = el.name
                if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    level = int(tag_name[1])
                    lines.append(f"\n{'#' * level} {text}\n")
                elif tag_name == "li":
                    lines.append(f"- {text}")
                elif tag_name == "pre":
                    lines.append(f"```\n{text}\n```")
                else:
                    lines.append(text)

            content = "\n".join(lines)
            # Collapse excessive whitespace
            content = re.sub(r"\n{3,}", "\n\n", content)

            if not content.strip():
                # Fallback: just get all text
                content = soup.get_text(separator="\n", strip=True)

            total_chars = len(content)
            truncated = content[:max_chars]
            return json.dumps(
                {
                    "content": truncated,
                    "char_count": total_chars,
                    "truncated": total_chars > max_chars,
                },
                ensure_ascii=False,
            )
        except requests.RequestException as e:
            if attempt == 2:
                return json.dumps({"error": f"Scrape failed: {e}"})
            time.sleep(2 ** attempt)

    return json.dumps({"error": "Scrape failed after retries"})


def _is_hf_dataset_url(url: str) -> bool:
    if not url:
        return False
    return "huggingface.co/datasets" in url or "huggingface.co/spaces" in url


# ---------------------------------------------------------------------------
# Tool registry  (name  ->  schema + implementation)
# ---------------------------------------------------------------------------

TOOL_REGISTRY: Dict[str, dict] = {
    "google_search": {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": (
                "Search the web via Google (Serper API). Returns organic "
                "results with title, link, snippet, and position."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {
                        "type": "string",
                        "description": "Search query string.",
                    },
                    "gl": {
                        "type": "string",
                        "description": "Region code (ISO 3166-1 alpha-2, e.g. 'us').",
                        "default": "us",
                    },
                    "hl": {
                        "type": "string",
                        "description": "Language code (ISO 639-1, e.g. 'en').",
                        "default": "en",
                    },
                    "num": {
                        "type": "integer",
                        "description": "Number of results (default 10).",
                        "default": 10,
                    },
                    "tbs": {
                        "type": "string",
                        "description": (
                            "Time filter: 'qdr:h' (hour), 'qdr:d' (day), "
                            "'qdr:w' (week), 'qdr:m' (month), 'qdr:y' (year)."
                        ),
                    },
                },
                "required": ["q"],
            },
        },
    },
    "scrape": {
        "type": "function",
        "function": {
            "name": "scrape",
            "description": (
                "Scrape / read a web page or other URL and return its "
                "content as readable text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to scrape.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    "submit_answer": {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": (
                "Submit your final answer. Call this once you have found the "
                "answer to the question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Your final answer to the question.",
                    },
                },
                "required": ["answer"],
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MiroFlowEnvArgs:
    """Configuration for MiroFlowEnv."""

    task_id: str
    task_question: str
    ground_truth: str = ""
    file_path: Optional[str] = None
    # API key (falls back to env var)
    serper_api_key: Optional[str] = None


class _DummyCommit:
    """Shim so code that accesses ``env.runtime.commit`` keeps working."""

    def get_patch(self, **kwargs) -> str:
        return ""


class _MiroFlowRuntime:
    """Minimal runtime shim (matches the subset of ``DockerRuntime`` that
    ``Agent.run()`` and ``Trajectory`` reference)."""

    def __init__(self, args: MiroFlowEnvArgs):
        self.docker_image = "miroflow/browsecomp"
        self.commit = _DummyCommit()
        self.ds: Dict[str, Any] = {
            "task_id": args.task_id,
            "problem_statement": args.task_question,
            "ground_truth": args.ground_truth,
        }

    def get_task_instruction(self) -> str:
        return self.ds["problem_statement"]

    def get_patch(self) -> str:
        return ""

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class MiroFlowEnv:
    """Gym-style environment backed by HTTP tool calls (Serper + free scraper).

    Follows the same contract as ``RepoEnv``:

        env = MiroFlowEnv(args)
        env.reset()
        env.add_commands(...)          # no-op
        obs, r, done, info = env.step(action)
        env.close()

    Actions are ``Action(function_name, parameters)`` where
    ``function_name`` is one of the registered tool names.
    """

    def __init__(
        self,
        args: MiroFlowEnvArgs,
        logger=None,
        step_timeout: int = 300,
        verbose: bool = True,
    ):
        if logger is None:
            self.logger = get_logger("MiroFlowEnv")
        else:
            self.logger = logger

        if not verbose:
            self.logger.setLevel(logging.CRITICAL)

        self.args = args
        self.done = False
        self.observation = None
        self.state = None
        self.step_timeout = step_timeout
        self.final_answer: Optional[str] = None

        # API key (explicit > env var)
        self.serper_api_key = args.serper_api_key or os.environ.get(
            "SERPER_API_KEY", ""
        )

        # Compatibility shim
        self.runtime = _MiroFlowRuntime(args)
        self.commands: list = []  # kept for ``format_command_docs`` compat

        self.logger.info(f"Initialized MiroFlowEnv for task: {args.task_id}")

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------

    def reset(self) -> str:
        self.observation = "Environment reset"
        self.state = None
        self.done = False
        self.final_answer = None
        return self.observation

    def add_commands(self, cmd_files: list = None):
        """No-op – tools are built-in HTTP calls, not bash command files."""
        pass

    def step(
        self,
        action: Action,
        timeout: int = None,
    ) -> Tuple[Observation, int, bool, Dict[str, Any]]:
        """Execute a tool call and return an Observation."""
        if not timeout:
            timeout = self.step_timeout

        # Handle submit / finish
        if action.function_name in ("submit_answer", "finish", "submit"):
            self.final_answer = action.parameters.get("answer", "")
            self.done = True
            obs = Observation("Answer submitted.", 0, action)
            return obs, 0, True, {"total_time": 0.0}

        output, error_code, elapsed = self._run_action(action, timeout)
        self.observation = Observation(output, error_code, action)
        return self.observation, 0, self.done, {"total_time": elapsed}

    def compute_reward(self, timeout: int = None) -> float:
        """BrowseComp evaluation uses an external LLM judge – see the
        run script for the verifier logic."""
        return 0

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _run_action(self, action: Action, timeout: int) -> Tuple[str, int, float]:
        if not action.function_name:
            return "", 0, 0.0

        name = action.function_name
        params = action.parameters
        start = time.time()

        try:
            if name == "google_search":
                output = _serper_search(
                    q=params.get("q", ""),
                    gl=params.get("gl", "us"),
                    hl=params.get("hl", "en"),
                    num=int(params.get("num", 10)),
                    tbs=params.get("tbs"),
                    page=params.get("page"),
                    api_key=self.serper_api_key,
                )
                return output, 0, time.time() - start

            elif name == "scrape":
                output = _free_scrape(
                    url=params.get("url", ""),
                )
                return output, 0, time.time() - start

            else:
                msg = (
                    f"Unknown tool '{name}'. "
                    f"Available: {list(TOOL_REGISTRY.keys())}"
                )
                self.logger.error(msg)
                return msg, -1, time.time() - start

        except Exception as e:
            return f"Tool '{name}' failed: {e}", -1, time.time() - start

    # ------------------------------------------------------------------
    # Tool definitions (OpenAI function-calling format)
    # ------------------------------------------------------------------

    def get_tool_definitions(self) -> List[dict]:
        """Return tool schemas compatible with ``litellm.completion(tools=...)``."""
        return list(TOOL_REGISTRY.values())

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def get_task_instruction(self) -> str:
        return self.args.task_question

    def get_stats(self) -> Dict[str, Any]:
        return {
            "task_id": self.args.task_id,
            "task_question": self.args.task_question,
            "ground_truth": self.args.ground_truth,
            "final_answer": self.final_answer,
        }
