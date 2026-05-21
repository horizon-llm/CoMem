"""
Action consistency reward for BrowseComp tasks.

BrowseComp tools:
  - google_search(q, gl?, hl?, num?, tbs?)  — web search
  - scrape(url)                              — read a web page
  - submit_answer(answer)                    — submit final answer

Reward is in [0, 10].  The overall structure mirrors the SWE-bench
action_consistency_reward but uses similarity functions tuned for
search queries, URLs, and short-form answers instead of code/paths.
"""

import re
import json
from difflib import SequenceMatcher
from typing import Dict, Tuple, Any
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Action / ActionParser — reused from swe_function_match_v3
# ---------------------------------------------------------------------------

_THOUGHT_RE = re.compile(r"(?s)(<think>.*?</think>)")
_ACTION_RE = re.compile(r"(?s)(<function=.*?</function>)")


class Action:
    def __init__(self, function_name: str, parameters: Dict[str, str], function_id: str = None):
        self.function_name = function_name
        self.parameters = parameters
        self.function_id = function_id

    @classmethod
    def from_string(cls, action_str: str) -> "Action":
        fn_match = re.search(r"<function\s*=\s*([^>]+)>", action_str)
        function_name = fn_match.group(1).strip() if fn_match else ""
        pattern = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"
        param_matches = re.findall(pattern, action_str, flags=re.DOTALL)
        params = {k.strip(): v.strip() for k, v in param_matches}
        return cls(function_name, params)

    def __str__(self) -> str:
        return self.to_xml_string()

    def __repr__(self) -> str:
        return f"Action(function_name={self.function_name!r}, parameters={self.parameters!r})"

    def to_xml_string(self) -> str:
        xml_str = f"<function={self.function_name}>\n"
        for k, v in self.parameters.items():
            xml_str += f"  <parameter={k}>{v}</parameter>\n"
        xml_str += "</function>"
        return xml_str

    def to_dict(self) -> Dict[str, object]:
        return {"function": self.function_name, "parameters": self.parameters}

    def is_empty(self) -> bool:
        return not self.function_name


class ActionParser:
    def parse(self, response, use_tool_call: bool = False, glm_model: bool = False):
        if use_tool_call:
            if glm_model:
                return self._parse_glm_tool_call(response)
            return self._parse_native_tool_call(response)
        return self._parse_xml_markup(response)

    def _parse_xml_markup(self, response):
        if not response:
            return "", Action("", {})
        if isinstance(response, dict):
            response = response.get("content", "") or ""
        m_thought = _THOUGHT_RE.search(response)
        m_action = _ACTION_RE.search(response)
        thought = m_thought.group(1).strip() if m_thought else ""
        action = Action.from_string(m_action.group(1).strip()) if m_action else Action("", {})
        return thought, action

    def _parse_native_tool_call(self, response):
        try:
            if hasattr(response, 'choices'):
                thought = response.choices[0].message.content or ""
            elif isinstance(response, dict):
                thought = response.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            else:
                thought = ""
        except (AttributeError, IndexError, KeyError, TypeError):
            thought = ""
        try:
            if hasattr(response, 'choices'):
                tc = response.choices[0].message.tool_calls[0]
                fn = tc.function.name
                params = json.loads(tc.function.arguments)
            elif isinstance(response, dict):
                tcs = response.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
                if tcs:
                    fn = tcs[0].get("function", {}).get("name", "")
                    params = json.loads(tcs[0].get("function", {}).get("arguments", "{}"))
                else:
                    fn, params = "", {}
            else:
                fn, params = "", {}
            action = Action(fn, params)
        except (AttributeError, IndexError, KeyError, TypeError, json.JSONDecodeError):
            action = Action("", {})
        return thought, action

    def _parse_glm_tool_call(self, response):
        if response is None:
            return "", Action("", {})
        try:
            if hasattr(response, 'choices') and response.choices:
                message = response.choices[0].message
            elif isinstance(response, dict):
                choices = response.get("choices", [])
                message = choices[0].get("message") if choices else None
            else:
                message = None
        except (AttributeError, IndexError, KeyError, TypeError):
            message = None
        if message is None:
            return "", Action("", {})
        thought = ""
        try:
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                thought = message.reasoning_content
        except Exception:
            pass
        if not thought and message.content:
            thought = message.content
        try:
            fn = message.tool_calls[0].function.name
            params = json.loads(message.tool_calls[0].function.arguments)
            action = Action(fn, params)
        except Exception:
            action = Action("", {})
        return thought, action


def parse_action(response, use_tool_call: bool = False, glm_model: bool = False):
    return ActionParser().parse(response, use_tool_call=use_tool_call, glm_model=glm_model)


# ---------------------------------------------------------------------------
# Similarity primitives for BrowseComp
# ---------------------------------------------------------------------------

def _token_jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity (case-insensitive)."""
    ta = {t for t in re.split(r"\s+", a.strip().lower()) if t}
    tb = {t for t in re.split(r"\s+", b.strip().lower()) if t}
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _string_sim(a: str, b: str) -> float:
    """SequenceMatcher similarity (case-insensitive)."""
    a, b = a.strip(), b.strip()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ---- Search query similarity ----

def _query_sim(q_a: str, q_b: str) -> float:
    """
    Similarity for search queries.

    Queries express *intent* so we care about:
      - Keyword overlap (token Jaccard) — most important
      - Subsequence similarity (SequenceMatcher) — captures phrasing
      - Quoted-phrase overlap — exact phrases matter

    Returns in [0, 1].
    """
    a = q_a.strip().lower()
    b = q_b.strip().lower()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    token_sim = _token_jaccard(a, b)
    seq_sim = _string_sim(a, b)

    # Check for quoted phrase overlap (e.g. "exact match" in both queries)
    quotes_a = set(re.findall(r'"([^"]+)"', a))
    quotes_b = set(re.findall(r'"([^"]+)"', b))
    if quotes_a and quotes_b:
        quote_overlap = len(quotes_a & quotes_b) / len(quotes_a | quotes_b)
    elif not quotes_a and not quotes_b:
        quote_overlap = 1.0  # neither uses quotes — neutral
    else:
        quote_overlap = 0.5  # one uses quotes, other doesn't

    return 0.50 * token_sim + 0.30 * seq_sim + 0.20 * quote_overlap


# ---- URL similarity ----

def _url_sim(url_a: str, url_b: str) -> float:
    """
    Similarity for URLs being scraped.

    We compare:
      - Domain (must match for high score)
      - Path (component overlap)
      - Exact match bonus

    Returns in [0, 1].
    """
    a = url_a.strip().rstrip("/")
    b = url_b.strip().rstrip("/")
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    try:
        pa = urlparse(a if "://" in a else "https://" + a)
        pb = urlparse(b if "://" in b else "https://" + b)
    except Exception:
        return _string_sim(a, b)

    # Domain comparison (strip www.)
    domain_a = pa.netloc.lower().lstrip("www.")
    domain_b = pb.netloc.lower().lstrip("www.")
    domain_match = 1.0 if domain_a == domain_b else 0.0

    # Path comparison
    parts_a = [p for p in pa.path.split("/") if p]
    parts_b = [p for p in pb.path.split("/") if p]
    if not parts_a and not parts_b:
        path_sim = 1.0
    elif not parts_a or not parts_b:
        path_sim = 0.0
    else:
        inter = len(set(parts_a) & set(parts_b))
        union = len(set(parts_a) | set(parts_b))
        path_sim = inter / union

    # Domain is critical — different domains = different pages
    if domain_match == 0.0:
        return 0.15 * path_sim  # cap at ~0.15 if domain differs

    return 0.50 * domain_match + 0.50 * path_sim


# ---- Answer similarity ----

_PUNCT_RE = re.compile(r"[^\w\s,]")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_answer(s: str) -> str:
    """Normalize an answer for comparison: lowercase, strip punctuation, collapse whitespace."""
    s = s.strip().lower()
    s = _PUNCT_RE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def _answer_sim(ans_a: str, ans_b: str) -> float:
    """
    Similarity for submitted answers.

    BrowseComp answers are short-form factual answers.  We need to handle:
      - Case insensitivity ("Paris" vs "paris")
      - Punctuation differences ("U.S.A." vs "USA")
      - Numerical precision ("3.14" vs "3.1416")
      - Comma-separated lists in any order ("A, B, C" vs "B, A, C")
      - Abbreviations ("Dr." vs "Doctor")

    Returns in [0, 1].
    """
    a = ans_a.strip()
    b = ans_b.strip()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    na = _normalize_answer(a)
    nb = _normalize_answer(b)
    if na == nb:
        return 1.0

    # Try numeric comparison
    try:
        fa, fb = float(na), float(nb)
        if fa == fb:
            return 1.0
        # Relative tolerance for numerical answers
        rel_diff = abs(fa - fb) / max(abs(fa), abs(fb), 1e-9)
        if rel_diff < 0.01:
            return 0.95
        if rel_diff < 0.05:
            return 0.8
    except ValueError:
        pass

    # Try comma-separated list comparison (order-independent)
    parts_a = sorted(s.strip() for s in na.split(",") if s.strip())
    parts_b = sorted(s.strip() for s in nb.split(",") if s.strip())
    if len(parts_a) > 1 or len(parts_b) > 1:
        if parts_a == parts_b:
            return 1.0
        # Set overlap for lists
        set_a, set_b = set(parts_a), set(parts_b)
        if set_a and set_b:
            list_sim = len(set_a & set_b) / len(set_a | set_b)
            # Also check sequence similarity of the joined forms
            joined_sim = _string_sim(", ".join(parts_a), ", ".join(parts_b))
            return max(list_sim, joined_sim)

    # SequenceMatcher on normalized forms
    return SequenceMatcher(None, na, nb).ratio()


# ---------------------------------------------------------------------------
# Main reward functions
# ---------------------------------------------------------------------------

# Per-function parameter weights for BrowseComp tools
_BROWSECOMP_KEY_WEIGHTS = {
    # google_search
    "q": 1.0,       # search query — the core parameter
    "gl": 0.2,      # region code — minor
    "hl": 0.2,      # language code — minor
    "num": 0.1,     # result count — negligible
    "tbs": 0.3,     # time filter — somewhat relevant
    # scrape
    "url": 1.0,     # the URL being scraped
    # submit_answer
    "answer": 1.0,  # the final answer
}

# Which similarity function to use for each parameter
_PARAM_SIM_DISPATCH = {
    "q": _query_sim,
    "url": _url_sim,
    "answer": _answer_sim,
}


def action_consistency_reward(action_a: Action, action_b: Action) -> float:
    """
    BrowseComp action consistency reward in [0, 10].

    Compares two actions (predicted vs expected) by:
      1. Function name match (40% weight)
      2. Parameter similarity using tool-specific comparators (60% weight)

    For google_search: query similarity (keyword overlap + phrasing)
    For scrape: URL similarity (domain + path)
    For submit_answer: answer similarity (normalized text + numeric + list handling)
    """
    fn_a = action_a.function_name.strip().lower() if action_a.function_name else None
    fn_b = action_b.function_name.strip().lower() if action_b.function_name else None

    if not fn_a or not fn_b:
        return 0.0

    p_a = {k.strip().lower(): v for k, v in action_a.parameters.items()}
    p_b = {k.strip().lower(): v for k, v in action_b.parameters.items()}

    fn_match = 1.0 if fn_a == fn_b else 0.0
    fn_weight = 0.40

    keys = set(p_a.keys()) | set(p_b.keys())
    if not keys:
        param_score = 1.0
    else:
        num = 0.0
        den = 0.0
        for k in keys:
            w = _BROWSECOMP_KEY_WEIGHTS.get(k, 0.3)
            va = str(p_a.get(k, ""))
            vb = str(p_b.get(k, ""))
            sim_fn = _PARAM_SIM_DISPATCH.get(k, _string_sim)
            sim = sim_fn(va, vb)
            num += w * sim
            den += w
        param_score = num / den if den > 0 else 0.0

    score01 = fn_weight * fn_match + (1.0 - fn_weight) * param_score

    # Cap score when function names don't match
    if fn_match == 0.0:
        score01 = min(score01, 0.35)

    return 10.0 * max(0.0, min(1.0, score01))


def action_consistency_reward_detailed(
    action_a: Action, action_b: Action
) -> Tuple[float, Dict[str, Any]]:
    """
    Same as action_consistency_reward but returns a detailed breakdown dict.
    """
    fn_a = action_a.function_name.strip().lower() if action_a.function_name else None
    fn_b = action_b.function_name.strip().lower() if action_b.function_name else None

    details: Dict[str, Any] = {}

    if not fn_a or not fn_b:
        details["function_match"] = 0.0
        details["param_score"] = 0.0
        details["error"] = 1.0 if (not fn_a and not fn_b) else 0.5
        return 0.0, details

    p_a = {k.strip().lower(): v for k, v in action_a.parameters.items()}
    p_b = {k.strip().lower(): v for k, v in action_b.parameters.items()}

    fn_match = 1.0 if fn_a == fn_b else 0.0
    details["function_match"] = fn_match
    details["function_a"] = fn_a
    details["function_b"] = fn_b

    fn_weight = 0.40

    keys = set(p_a.keys()) | set(p_b.keys())
    param_details: Dict[str, float] = {}

    if not keys:
        param_score = 1.0
    else:
        num = 0.0
        den = 0.0
        for k in keys:
            w = _BROWSECOMP_KEY_WEIGHTS.get(k, 0.3)
            va = str(p_a.get(k, ""))
            vb = str(p_b.get(k, ""))
            sim_fn = _PARAM_SIM_DISPATCH.get(k, _string_sim)
            sim = sim_fn(va, vb)
            param_details[k] = sim
            num += w * sim
            den += w
        param_score = num / den if den > 0 else 0.0

    details["param_score"] = param_score
    details["param_details"] = param_details

    score01 = fn_weight * fn_match + (1.0 - fn_weight) * param_score
    if fn_match == 0.0:
        score01 = min(score01, 0.35)

    details["final_score"] = score01
    reward = 10.0 * max(0.0, min(1.0, score01))

    return reward, details
