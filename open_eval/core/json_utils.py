import json
from typing import Any, Dict


def extract_first_json_object(text: str) -> str:
    if not text:
        raise ValueError("empty text")

    s = text.strip()
    start = s.find("{")
    if start < 0:
        raise ValueError("no '{' found")

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]

    raise ValueError("no complete JSON object found (unbalanced braces)")


def robust_json_loads(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    js = extract_first_json_object(text)
    return json.loads(js)
