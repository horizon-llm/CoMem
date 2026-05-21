import re
import os
import json
import shlex
from difflib import SequenceMatcher
from typing import Dict, Tuple, Optional, Set, Any

# Pattern for thought blocks
_THOUGHT_RE = re.compile(r"(?s)(<think>.*?</think>)")
# Pattern for action blocks
_ACTION_RE = re.compile(r"(?s)(<function=.*?</function>)")


class Action:
    """
    Represents an action with:
      - function_name (e.g. 'file_editor')
      - parameters    (a dictionary of parameter_name -> value)

    Provides methods:
      - from_string(...) -> create Action from XML-like string
      - to_dict()        -> returns a JSON-like dict
      - to_bashcmd()     -> returns a string representing an equivalent bash command
      - to_xml_string()  -> returns XML-like string representation
    """

    def __init__(
        self, function_name: str, parameters: Dict[str, str], function_id: str = None
    ):
        self.function_name = function_name
        self.parameters = parameters
        self.function_id = function_id

    @classmethod
    def from_string(cls, action_str: str) -> "Action":
        """
        Parses a string of the form:

          <function=FUNCTION_NAME>
            <parameter=KEY>VALUE</parameter>
            ...
          </function>

        and returns an Action object.

        For example:
          <function=file_editor>
            <parameter=command>view</parameter>
            <parameter=path>./sympy/tensor/array/dense_ndim_array.py</parameter>
            <parameter=concise>True</parameter>
          </function>

        yields an Action with:
          function_name = "file_editor"
          parameters = {
            "command":  "view",
            "path":     "./sympy/tensor/array/dense_ndim_array.py",
            "concise":  "True"
          }
        """
        # Extract the function name: <function=...>
        fn_match = re.search(r"<function\s*=\s*([^>]+)>", action_str)
        function_name = fn_match.group(1).strip() if fn_match else ""

        # Extract parameters of the form: <parameter=KEY>VALUE</parameter>
        # DOTALL allows the captured VALUE to span multiple lines
        pattern = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"
        param_matches = re.findall(pattern, action_str, flags=re.DOTALL)

        params = {}
        for param_key, param_value in param_matches:
            param_key = param_key.strip()
            param_value = param_value.strip()
            params[param_key] = param_value

        return cls(function_name, params)

    def __str__(self) -> str:
        return self.to_xml_string()

    def __repr__(self) -> str:
        return f"Action(function_name={self.function_name!r}, parameters={self.parameters!r})"

    def to_xml_string(self) -> str:
        """
        Returns an XML-like string representation of this action.

        Example:
          <function=file_editor>
            <parameter=command>view</parameter>
            <parameter=path>./sympy/tensor/array/dense_ndim_array.py</parameter>
            <parameter=concise>True</parameter>
          </function>
        """
        xml_str = f"<function={self.function_name}>\n"

        for param_key, param_value in self.parameters.items():
            xml_str += f"  <parameter={param_key}>{param_value}</parameter>\n"

        xml_str += "</function>"
        return xml_str

    def to_dict(self) -> Dict[str, object]:
        """
        Returns a JSON-like dictionary representation of this action.

        Example:
          {
            "function": "file_editor",
            "parameters": {
              "command": "view",
              "path": "./sympy/tensor/array/dense_ndim_array.py",
              "concise": "True"
            }
          }
        """
        return {"function": self.function_name, "parameters": self.parameters}

    def to_bashcmd(self) -> str:
        """
        Converts this action into a Bash command string.

        Examples:
          If function_name == "execute_bash" and parameters = {
             "command": "search_dir",
             "search_term": "foo"
          }
          then this returns:
            execute_bash search_dir --search_term 'foo'

          If function_name == "file_editor" and parameters = {
             "command": "view",
             "path": "./some/path.py",
             "concise": "True"
          }
          then this returns:
            file_editor view --path './some/path.py' --concise 'True'
        """
        if not self.function_name:
            return ""
        elif self.function_name == "finish" or self.function_name == "submit":
            return "echo '<<<Finished>>>'"

        cmd_parts = [shlex.quote(self.function_name)]

        base_command = self.parameters.get("command")
        if base_command is not None:
            cmd_parts.append(shlex.quote(base_command))

        for param_key, param_value in self.parameters.items():
            if param_key == "command":
                continue
            param_value_quoted = shlex.quote(str(param_value))
            cmd_parts.append(f"--{param_key}")
            cmd_parts.append(param_value_quoted)

        return " ".join(cmd_parts)

    def is_empty(self) -> bool:
        """Returns True if this action has no function name."""
        return not self.function_name


class ActionParser:
    """
    Parser for extracting actions from LLM responses.

    Supports two modes (specified via use_tool_call argument):
    1. XML markup mode (use_tool_call=False): For LLMs that don't natively support tool calls
       - Parses <function=...><parameter=...>...</parameter></function> format
    2. Native tool call mode (use_tool_call=True): For LLMs with native tool call support
       - Extracts function name and arguments from the response object
    """

    def parse(self, response: Any, use_tool_call: bool = False, glm_model: bool = False) -> Tuple[str, Action]:
        """
        Parse the response and extract thought and action.

        Args:
            response: Either a string (for XML mode) or a response object
                     with choices[0].message (for native tool call mode)
            use_tool_call: If True, parse as native tool call response.
                          If False, parse as XML markup string.

        Returns:
            Tuple of (thought, Action)
        """
        if use_tool_call:
            if glm_model:
                return self._parse_glm_tool_call(response)
            else:
                return self._parse_native_tool_call(response)
        else:
            return self._parse_xml_markup(response)

    def _parse_xml_markup(self, response: str) -> Tuple[str, Action]:
        """
        Parse XML-style markup from LLM response.

        Extracts:
        - thought: content in <think>...</think> block
        - action: the entire first <function=...></function> block

        Args:
            response: The raw string response from the LLM

        Returns:
            Tuple of (thought, Action)
        """
        if not response:
            return "", Action(function_name="", parameters={})

        # Handle case where response might be a dict with 'content' key
        if isinstance(response, dict):
            response = response.get("content", "") or ""

        match_thought = _THOUGHT_RE.search(response)
        match_action = _ACTION_RE.search(response)

        if match_thought:
            thought = match_thought.group(1).strip()
        else:
            thought = ""

        if match_action:
            action_str = match_action.group(1).strip()
            action = Action.from_string(action_str)
        else:
            action = Action(function_name="", parameters={})

        return thought, action

    def _parse_native_tool_call(self, response: Any) -> Tuple[str, Action]:
        """
        Parse native tool call from LLM response object.

        Expected response structure (OpenAI-style):
        {
            "choices": [{
                "message": {
                    "content": "...",  # thought/reasoning
                    "tool_calls": [{
                        "function": {
                            "name": "...",
                            "arguments": "{...}"  # JSON string
                        }
                    }]
                }
            }]
        }

        Args:
            response: The response object from the LLM API

        Returns:
            Tuple of (thought, Action)
        """
        # Extract thought from message content
        try:
            if hasattr(response, 'choices'):
                thought = response.choices[0].message.content or ""
            elif isinstance(response, dict):
                thought = response.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            else:
                thought = ""
        except (AttributeError, IndexError, KeyError, TypeError):
            thought = ""

        # Extract action from tool_calls
        try:
            if hasattr(response, 'choices'):
                tool_call = response.choices[0].message.tool_calls[0]
                function_name = tool_call.function.name
                parameters = json.loads(tool_call.function.arguments)
            elif isinstance(response, dict):
                tool_calls = response.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
                if tool_calls:
                    tool_call = tool_calls[0]
                    function_name = tool_call.get("function", {}).get("name", "")
                    parameters = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                else:
                    function_name = ""
                    parameters = {}
            else:
                function_name = ""
                parameters = {}
            action = Action(function_name=function_name, parameters=parameters)
        except (AttributeError, IndexError, KeyError, TypeError, json.JSONDecodeError):
            action = Action(function_name="", parameters={})

        return thought, action
    
    def _parse_glm_tool_call(self, response: Any) -> Tuple[str, Action]:
        """
        Custom parser for GLM-4.7 models which return reasoning in
        'reasoning' or 'reasoning_content' fields on the tool_call object.
        """
        # Handle None or invalid response
        if response is None:
            return "", Action(function_name="", parameters={})

        # Safely extract message from response
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

        # If message is None, return empty results
        if message is None:
            return "", Action(function_name="", parameters={})

        # GLM-4.7 puts reasoning on the tool_call object, not the message
        thought = ""
        try:
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                thought = message.reasoning_content
        except:
            pass

        # Fall back to message content if no reasoning found in tool_call
        if not thought and message.content:
            thought = message.content

        try:
            function_name = message.tool_calls[0].function.name
            parameters = json.loads(message.tool_calls[0].function.arguments)
            action = Action(function_name=function_name, parameters=parameters)
        except:
            action = Action(function_name="", parameters={})

        return thought, action


def parse_action(response: Any, use_tool_call: bool = False, glm_model: bool = False) -> Tuple[str, Action]:
    """
    Convenience function to parse an LLM response and extract thought and action.

    Args:
        response: The LLM response. Either a string (for XML mode) or a response
                 object with choices[0].message (for native tool call mode).
        use_tool_call: If True, parse as native tool call response (e.g., OpenAI).
                      If False, parse as XML markup string.

    Returns:
        Tuple of (thought, Action)

    Examples:
        # For LLMs without native tool calls (XML markup):
        thought, action = parse_action(response_text, use_tool_call=False)

        # For LLMs with native tool calls (e.g., OpenAI):
        thought, action = parse_action(response_obj, use_tool_call=True)
    """
    parser = ActionParser()
    return parser.parse(response, use_tool_call=use_tool_call, glm_model=glm_model)

# Tokenizer for code: matches identifiers, numbers, operators, etc.
_CODE_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+|[^\s\w]")


def _norm_bool(s: str) -> Optional[bool]:
    t = s.strip().lower()
    if t in {"true", "1", "yes"}:
        return True
    if t in {"false", "0", "no"}:
        return False
    return None


def _path_sim(path_a: str, path_b: str) -> float:
    """
    Similarity for file paths: based on normalized common prefix length and
    path component overlap. Returns in [0, 1].
    """
    a = path_a.strip()
    b = path_b.strip()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    # Common string prefix (helps /testbed/pandas/... cases)
    pref = os.path.commonprefix([a, b])
    pref_ratio = len(pref) / max(len(a), len(b))

    # Component overlap (helps when prefixes differ slightly)
    a_parts = [p for p in a.split("/") if p]
    b_parts = [p for p in b.split("/") if p]
    if not a_parts or not b_parts:
        comp_ratio = 0.0
    else:
        inter = len(set(a_parts) & set(b_parts))
        union = len(set(a_parts) | set(b_parts))
        comp_ratio = inter / union if union else 0.0

    return 0.7 * pref_ratio + 0.3 * comp_ratio


def _token_jaccard(a: str, b: str) -> float:
    """
    Token Jaccard similarity (good for search_term like "def to_markdown").
    Returns in [0, 1].
    """
    ta = {t for t in re.split(r"\s+", a.strip().lower()) if t}
    tb = {t for t in re.split(r"\s+", b.strip().lower()) if t}
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _string_sim(a: str, b: str) -> float:
    """
    Generic string similarity in [0, 1].
    """
    a = a.strip()
    b = b.strip()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    ba = _norm_bool(a)
    bb = _norm_bool(b)
    if ba is not None and bb is not None:
        return 1.0 if ba == bb else 0.0

    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _normalize_code(s: str) -> str:
    """
    Normalize code for comparison:
    - Strip leading/trailing whitespace from each line
    - Remove empty lines
    - Collapse to single string
    """
    lines = [line.strip() for line in s.strip().splitlines()]
    return "\n".join(line for line in lines if line)


def _get_code_tokens(s: str) -> Set[str]:
    """
    Extract code tokens (identifiers, numbers, operators) from a string.
    """
    return set(_CODE_TOKEN_RE.findall(s.lower()))


def _code_token_sim(a: str, b: str) -> float:
    """
    Token-based Jaccard similarity for code.
    Handles variable renaming and minor structural differences.
    """
    tokens_a = _get_code_tokens(a)
    tokens_b = _get_code_tokens(b)

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


def _code_line_sim(a: str, b: str) -> float:
    """
    Line-based Jaccard similarity for code.
    Compares normalized lines as sets.
    """
    lines_a = set(_normalize_code(a).splitlines())
    lines_b = set(_normalize_code(b).splitlines())

    if not lines_a and not lines_b:
        return 1.0
    if not lines_a or not lines_b:
        return 0.0

    intersection = len(lines_a & lines_b)
    union = len(lines_a | lines_b)
    return intersection / union


def _code_seq_sim(a: str, b: str) -> float:
    """
    SequenceMatcher similarity on normalized code.
    """
    na = _normalize_code(a)
    nb = _normalize_code(b)

    if na == nb:
        return 1.0
    if not na or not nb:
        return 0.0

    # For very long strings, SequenceMatcher can be slow
    # Use a length limit and fall back to token-based for long strings
    MAX_LEN = 5000
    if len(na) > MAX_LEN or len(nb) > MAX_LEN:
        # Fall back to comparing first/last portions + token sim
        na_truncated = na[:MAX_LEN//2] + na[-MAX_LEN//2:]
        nb_truncated = nb[:MAX_LEN//2] + nb[-MAX_LEN//2:]
        return SequenceMatcher(None, na_truncated, nb_truncated).ratio()

    return SequenceMatcher(None, na, nb).ratio()


def _code_sim(a: str, b: str) -> float:
    """
    Combined code similarity using multiple signals:
    - Normalized SequenceMatcher (handles formatting differences)
    - Token Jaccard (handles variable renaming, robust to reordering)
    - Line overlap (handles structural similarity)

    Returns in [0, 1].
    """
    a = a.strip()
    b = b.strip()

    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    # Check normalized equality first
    na = _normalize_code(a)
    nb = _normalize_code(b)
    if na == nb:
        return 1.0

    # Compute individual similarities
    seq_sim = _code_seq_sim(a, b)
    token_sim = _code_token_sim(a, b)
    line_sim = _code_line_sim(a, b)

    # Weighted combination:
    # - seq_sim: captures exact character-level similarity
    # - token_sim: robust to variable renaming, formatting
    # - line_sim: captures structural similarity (same lines present)
    #
    # Weights tuned for comparing model outputs where:
    # - We care about targeting the same code location (high seq weight)
    # - We want some tolerance for minor differences (token weight)
    # - Line overlap helps with multi-line blocks (line weight)
    return 0.45 * seq_sim + 0.35 * token_sim + 0.20 * line_sim


def _code_sim_strict(a: str, b: str) -> float:
    """
    Stricter code similarity for old_str matching.
    When comparing old_str, we want models to target the exact same code location.
    """
    a = a.strip()
    b = b.strip()

    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    na = _normalize_code(a)
    nb = _normalize_code(b)
    if na == nb:
        return 1.0

    # Stricter weighting - more emphasis on exact sequence match
    seq_sim = _code_seq_sim(a, b)
    token_sim = _code_token_sim(a, b)

    return 0.7 * seq_sim + 0.3 * token_sim


def _code_sim_lenient(a: str, b: str) -> float:
    """
    More lenient code similarity for new_str/file_text matching.
    When comparing new code, we allow more variation in implementation details.
    """
    a = a.strip()
    b = b.strip()

    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    na = _normalize_code(a)
    nb = _normalize_code(b)
    if na == nb:
        return 1.0

    # More lenient weighting - more emphasis on tokens and structure
    seq_sim = _code_seq_sim(a, b)
    token_sim = _code_token_sim(a, b)
    line_sim = _code_line_sim(a, b)

    return 0.30 * seq_sim + 0.45 * token_sim + 0.25 * line_sim


def _cmd_sim(a: str, b: str) -> float:
    """
    Similarity for bash commands (execute_bash cmd parameter).

    Compares:
    - The base command (python3, cd, ls, etc.)
    - Arguments and file paths
    - Handles variations like 'python' vs 'python3'

    Returns in [0, 1].
    """
    a = a.strip()
    b = b.strip()

    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    # Split into command parts
    parts_a = a.split()
    parts_b = b.split()

    if not parts_a or not parts_b:
        return 0.0

    # Compare base command (first part)
    cmd_a = parts_a[0].lower()
    cmd_b = parts_b[0].lower()

    # Handle common command variations
    cmd_aliases = {
        "python3": "python",
        "python2": "python",
        "pip3": "pip",
        "pip2": "pip",
    }
    cmd_a_norm = cmd_aliases.get(cmd_a, cmd_a)
    cmd_b_norm = cmd_aliases.get(cmd_b, cmd_b)

    cmd_match = 1.0 if cmd_a_norm == cmd_b_norm else 0.0

    # Compare arguments (remaining parts)
    args_a = parts_a[1:] if len(parts_a) > 1 else []
    args_b = parts_b[1:] if len(parts_b) > 1 else []

    if not args_a and not args_b:
        args_sim = 1.0
    elif not args_a or not args_b:
        args_sim = 0.0
    else:
        # Use Jaccard for argument similarity
        set_a = set(args_a)
        set_b = set(args_b)
        args_sim = len(set_a & set_b) / len(set_a | set_b)

        # Also consider path similarity for file arguments
        paths_a = [p for p in args_a if '/' in p or p.endswith('.py') or p.endswith('.sh')]
        paths_b = [p for p in args_b if '/' in p or p.endswith('.py') or p.endswith('.sh')]

        if paths_a and paths_b:
            # Compare the primary path (usually the script being run)
            path_sim = _path_sim(paths_a[0], paths_b[0])
            args_sim = 0.5 * args_sim + 0.5 * path_sim

    # Weight: command match is important, but arguments matter too
    return 0.4 * cmd_match + 0.6 * args_sim


def action_consistency_reward(action_a: Action, action_b: Action) -> float:
    """
    Reward in [0, 10] measuring how consistent two actions are.
    10 = identical intent; 0 = totally different / empty actions.

    Args:
        action_a: First Action object (already parsed by user)
        action_b: Second Action object (already parsed by user)

    Returns:
        Float reward in [0, 10]

    Compares:
    - Function name (must match for high score)
    - Parameters with appropriate similarity functions:
      - path: path-aware similarity
      - command: exact match preferred
      - old_str: strict code similarity (targeting same location)
      - new_str/file_text: lenient code similarity (similar changes)
      - search_term: token-based similarity
      - Others: generic string similarity

    Example:
        # User parses actions themselves using their preferred method:
        _, action_a = parse_action(response_a, use_tool_call=False)
        _, action_b = parse_action(response_b, use_tool_call=True)
        reward = action_consistency_reward(action_a, action_b)
    """
    # Extract function names (normalize to lowercase for comparison)
    fn_a = action_a.function_name.strip().lower() if action_a.function_name else None
    fn_b = action_b.function_name.strip().lower() if action_b.function_name else None

    # If either action is empty / unparsable
    if not fn_a or not fn_b:
        return 0.0

    # Get parameters (normalize keys to lowercase)
    p_a = {k.strip().lower(): v for k, v in action_a.parameters.items()}
    p_b = {k.strip().lower(): v for k, v in action_b.parameters.items()}

    # 1) Function match is the biggest factor
    fn_match = 1.0 if fn_a == fn_b else 0.0
    fn_weight = 0.40  # 40% of the score

    # 2) Parameter similarity (weighted by key importance)
    key_weights = {
        "command": 1.2,      # file_editor command type is critical
        "cmd": 1.0,          # execute_bash command
        "path": 1.0,         # file path is very important
        "old_str": 1.0,      # what we're replacing matters
        "new_str": 0.9,      # the replacement content
        "file_text": 0.8,    # file content for create
        "search_term": 0.9,  # search queries
        "view_range": 0.6,   # view range is less critical
        "concise": 0.5,      # minor parameter
    }

    keys = set(p_a.keys()) | set(p_b.keys())
    if not keys:
        param_score = 1.0  # no params on either side
    else:
        num = 0.0
        den = 0.0
        for k in keys:
            w = key_weights.get(k, 0.7)  # default weight for unknown params
            va = str(p_a.get(k, ""))
            vb = str(p_b.get(k, ""))

            # Choose similarity function based on parameter type
            if k == "path":
                sim = _path_sim(va, vb)
            elif k == "command":
                # file_editor command should match exactly or very closely
                sim = 1.0 if va.lower() == vb.lower() else 0.2
            elif k == "cmd":
                # execute_bash command - use specialized bash command similarity
                sim = _cmd_sim(va, vb)
            elif k == "old_str":
                # Strict matching - models should target same code location
                sim = _code_sim_strict(va, vb)
            elif k in ("new_str", "file_text"):
                # Lenient matching - allow variation in implementation
                sim = _code_sim_lenient(va, vb)
            elif k == "search_term":
                # Token overlap for search terms
                sim = 0.6 * _token_jaccard(va, vb) + 0.4 * _string_sim(va, vb)
            else:
                sim = _string_sim(va, vb)

            num += w * sim
            den += w
        param_score = num / den if den > 0 else 0.0

    # 3) Combine function match and parameter similarity
    score01 = fn_weight * fn_match + (1.0 - fn_weight) * param_score

    # 4) Penalty if function doesn't match (cap the score)
    if fn_match == 0.0:
        score01 = min(score01, 0.35)

    # Map to 0–10
    return 10.0 * max(0.0, min(1.0, score01))


def action_consistency_reward_detailed(
    action_a: Action, action_b: Action
) -> Tuple[float, Dict[str, Any]]:
    """
    Same as action_consistency_reward but also returns detailed breakdown.

    Args:
        action_a: First Action object (already parsed by user)
        action_b: Second Action object (already parsed by user)

    Returns:
        (reward, details_dict) where details_dict contains:
        - 'function_match': 1.0 or 0.0
        - 'function_a': function name from action_a
        - 'function_b': function name from action_b
        - 'param_score': overall parameter similarity
        - 'param_details': dict of per-parameter similarities
        - 'final_score': score in [0, 1] before scaling to [0, 10]

    Example:
        _, action_a = parse_action(response_a, use_tool_call=False)
        _, action_b = parse_action(response_b, use_tool_call=True)
        reward, details = action_consistency_reward_detailed(action_a, action_b)
    """
    # Extract function names (normalize to lowercase for comparison)
    fn_a = action_a.function_name.strip().lower() if action_a.function_name else None
    fn_b = action_b.function_name.strip().lower() if action_b.function_name else None

    details: Dict[str, Any] = {}

    if not fn_a or not fn_b:
        details["function_match"] = 0.0
        details["param_score"] = 0.0
        details["error"] = 1.0 if (not fn_a and not fn_b) else 0.5
        return 0.0, details

    # Get parameters (normalize keys to lowercase)
    p_a = {k.strip().lower(): v for k, v in action_a.parameters.items()}
    p_b = {k.strip().lower(): v for k, v in action_b.parameters.items()}

    fn_match = 1.0 if fn_a == fn_b else 0.0
    details["function_match"] = fn_match
    details["function_a"] = fn_a
    details["function_b"] = fn_b

    fn_weight = 0.40

    key_weights = {
        "command": 1.2,
        "cmd": 1.0,
        "path": 1.0,
        "old_str": 1.0,
        "new_str": 0.9,
        "file_text": 0.8,
        "search_term": 0.9,
        "view_range": 0.6,
        "concise": 0.5,
    }

    keys = set(p_a.keys()) | set(p_b.keys())
    param_details: Dict[str, float] = {}

    if not keys:
        param_score = 1.0
    else:
        num = 0.0
        den = 0.0
        for k in keys:
            w = key_weights.get(k, 0.7)
            va = str(p_a.get(k, ""))
            vb = str(p_b.get(k, ""))

            if k == "path":
                sim = _path_sim(va, vb)
            elif k == "command":
                sim = 1.0 if va.lower() == vb.lower() else 0.2
            elif k == "cmd":
                sim = _cmd_sim(va, vb)
            elif k == "old_str":
                sim = _code_sim_strict(va, vb)
            elif k in ("new_str", "file_text"):
                sim = _code_sim_lenient(va, vb)
            elif k == "search_term":
                sim = 0.6 * _token_jaccard(va, vb) + 0.4 * _string_sim(va, vb)
            else:
                sim = _string_sim(va, vb)

            param_details[k] = sim
            num += w * sim
            den += w
        param_score = num / den if den > 0 else 0.0

    details["param_score"] = param_score
    details["param_details"] = param_details

    score01 = fn_weight * fn_match + (1.0 - fn_weight) * param_score
    if fn_match == 0.0:
        score01 = min(score01, 0.35)

    reward = 10.0 * max(0.0, min(1.0, score01))
    details["final_score"] = score01

    return reward, details
