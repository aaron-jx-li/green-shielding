"""Semantic matcher using the Vertex Gemini judge (local; not open_eval)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from management_reasoning.eval.judge import call_json_judge
from management_reasoning.eval.prompts import SEM_MATCH_BATCH_SYSTEM
from management_reasoning.eval.text import fuzzy_match, normalize_text


class SemanticMatcher:
    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        cache_path: Optional[str] = None,
        client: Any = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        thinking_level: str = "HIGH",
    ):
        self.model = model
        self.temperature = temperature
        self.cache_path = cache_path
        self.client = client
        self.project = project
        self.location = location
        self.thinking_level = thinking_level
        self.cache: Dict[str, Dict[str, Any]] = {}
        if cache_path:
            self._load_cache(cache_path)

    @staticmethod
    def _key(a: str, b: str) -> str:
        a_n = normalize_text(a)
        b_n = normalize_text(b)
        if a_n <= b_n:
            return f"{a_n}|||{b_n}"
        return f"{b_n}|||{a_n}"

    def _load_cache(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                self.cache = obj
                print(f"[sem_match] Loaded cache entries: {len(self.cache)} from {path}")
        except FileNotFoundError:
            print(f"[sem_match] Cache file not found (will create): {path}")
        except Exception as e:
            print(f"[sem_match] Failed to load cache {path}: {e}")

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            print(f"[sem_match] Saved cache entries: {len(self.cache)} to {self.cache_path}")
        except Exception as e:
            print(f"[sem_match] Failed to save cache {self.cache_path}: {e}")

    def batch_match_pairs(
        self,
        pairs: List[Tuple[str, str]],
        max_pairs_per_call: int = 50,
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        to_judge: List[Tuple[str, str, str]] = []

        for a, b in pairs:
            a = (a or "").strip()
            b = (b or "").strip()
            k = self._key(a, b)

            if not a or not b:
                info = {"match": False, "relation": "different", "note": "empty"}
                results[k] = info
                self.cache[k] = info
                continue

            if fuzzy_match(a, b):
                info = {"match": True, "relation": "same", "note": "string_match"}
                results[k] = info
                self.cache[k] = info
                continue

            if k in self.cache:
                results[k] = self.cache[k]
            else:
                to_judge.append((a, b, k))

        i = 0
        while i < len(to_judge):
            chunk = to_judge[i : i + max_pairs_per_call]
            i += max_pairs_per_call

            stack = [chunk]
            while stack:
                sub = stack.pop()
                payload = [{"dx_a": a, "dx_b": b} for (a, b, _) in sub]
                user_prompt = (
                    "PAIRS:\n" + json.dumps(payload, ensure_ascii=False) + "\nReturn STRICT JSON."
                )

                try:
                    obj = call_json_judge(
                        model=self.model,
                        system_prompt=SEM_MATCH_BATCH_SYSTEM,
                        user_prompt=user_prompt,
                        temperature=self.temperature,
                        client=self.client,
                        project=self.project,
                        location=self.location,
                        thinking_level=self.thinking_level,
                    )
                    matches = obj.get("matches", [])
                    if not isinstance(matches, list):
                        raise ValueError("missing/invalid 'matches' list")
                except Exception as e:
                    if len(sub) <= 1:
                        (_a, _b, k) = sub[0]
                        out = {
                            "match": False,
                            "relation": "different",
                            "note": f"judge_fail:{type(e).__name__}",
                        }
                        results[k] = out
                        self.cache[k] = out
                        continue
                    mid = len(sub) // 2
                    stack.append(sub[mid:])
                    stack.append(sub[:mid])
                    continue

                if len(matches) < len(sub):
                    matches = matches + [False] * (len(sub) - len(matches))
                elif len(matches) > len(sub):
                    matches = matches[: len(sub)]

                for (_a, _b, k), m in zip(sub, matches):
                    out = {
                        "match": bool(m),
                        "relation": "same" if m else "different",
                        "note": "batch_bool",
                    }
                    results[k] = out
                    self.cache[k] = out

        return results
