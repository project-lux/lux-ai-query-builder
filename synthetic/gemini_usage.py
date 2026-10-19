"""
Shared token / cost ledger for all Gemini calls. Usage is accumulated in
gemini-usage.json across runs and processes; once the budget is spent, check()
raises BudgetExceeded so callers stop making requests.

Call check() before each request and record() / record_embedding() after it.
"""

import json
import os
import time
from contextlib import contextmanager

try:
    import fcntl
except ImportError:  # Windows: no cross-process locking
    fcntl = None

LEDGER_FN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini-usage.json")
INPUT_COST = 0.75 / 1_000_000  # $ per input token
OUTPUT_COST = 3.75 / 1_000_000  # $ per output token, including thinking
BUDGET = 10.0  # $


class BudgetExceeded(Exception):
    pass


@contextmanager
def _ledger():
    """Locked read-modify-write of the ledger file."""
    with open(LEDGER_FN + ".lock", "w") as lock:
        if fcntl:
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            with open(LEDGER_FN) as fh:
                data = json.load(fh)
        except FileNotFoundError:
            data = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0,
                    "calls": 0, "cost": 0.0, "by_label": {}}
        yield data
        data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = LEDGER_FN + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, LEDGER_FN)


def spent():
    try:
        with open(LEDGER_FN) as fh:
            return json.load(fh)["cost"]
    except FileNotFoundError:
        return 0.0


def check():
    cost = spent()
    if cost >= BUDGET:
        raise BudgetExceeded(f"Gemini budget of ${BUDGET:.2f} used (${cost:.4f} spent)")


def _add(label, input_tokens, output_tokens, thinking_tokens):
    cost = input_tokens * INPUT_COST + (output_tokens + thinking_tokens) * OUTPUT_COST
    with _ledger() as data:
        for d in (data, data["by_label"].setdefault(
                label, {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0,
                        "calls": 0, "cost": 0.0})):
            d["input_tokens"] += input_tokens
            d["output_tokens"] += output_tokens
            d["thinking_tokens"] += thinking_tokens
            d["calls"] += 1
            d["cost"] = round(d["cost"] + cost, 6)
        return data["cost"]


def record(usage, label):
    """Record a generate_content response's usage_metadata."""
    if usage is None:
        return spent()
    return _add(
        label,
        usage.prompt_token_count or 0,
        usage.candidates_token_count or 0,
        usage.thoughts_token_count or 0,
    )


def record_embedding(resp, label):
    """Record an embed_content response, charged as input tokens."""
    tokens = sum(
        int(e.statistics.token_count) for e in resp.embeddings
        if e.statistics and e.statistics.token_count
    )
    return _add(label, tokens, 0, 0)
