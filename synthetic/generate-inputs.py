"""
Generate a large, varied set of natural language search requests for testing the query
processor.

A plan of slots is built up front from generation-topics.json and generation-axes.json:
each topic gets --per-topic slots, each with its own record type, complexity, features
and phrasing, balanced across the whole run (no two slots on a topic share a record type
and complexity). Each Gemini call covers half the topics of one discipline and writes
one request per slot, so previous output never needs to be fed back to the model.

The plan is saved next to the output; re-running only fills in slots that are missing,
so runs can be resumed. Output is JSONL, one request per line with its slot spec.

Usage: python generate-inputs.py [--out generated-inputs.jsonl] [--limit-calls N]
"""

import argparse
import json
import os
import random
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

import gemini_usage

with open("google_project.txt") as fh:
    gcp_name = fh.read().strip()

gemini_model = "gemini-3.8-flash"
gemini = genai.Client(vertexai=True, project=gcp_name, location="global")

batch_prompt = """LUX is Yale University's unified collections discovery platform, covering Yale's
libraries, museums and archives: books, journals, manuscripts, artworks, photographs,
maps, natural history specimens, archival collections, people, organizations, places,
concepts and events.

Imagine Yale students, each describing to a helpful librarian what they want to find in
LUX. All of them are working in the area of {discipline}. Below is a numbered list of
{n} slots. For each slot, write what that student says, following the slot's topic,
what they want to find, complexity, constraints and phrasing style.

Each request must be a natural language DESCRIPTION of what the student wants, as they
would say it out loud, not a list of search keywords. For example:
  BAD (keywords):  "medieval maps"
  GOOD (natural):  "I'm looking for maps that were made during the Middle Ages"

Complexity is the number of separate constraints in the request. "Constraints" lists
which kinds of constraint to include, using whatever specific values fit the topic.
If a topic and the kind of thing wanted seem an awkward match, find the most natural
angle (for example, for places and a laboratory science: where the research was done,
or where the specimens came from).

Keep every request realistic: something a real student would plausibly want and could
reasonably expect a university collection to have. Even detailed requests should not
pile up obscure, improbable details. Make requests on the same topic clearly different
from each other, picking different aspects, people, places, periods or examples. Use
ordinary everyday language. Do not use database field names, query syntax, boolean
operators, or data model terminology.

Slots:
{slots}

Return a JSON array with one object per slot: {{"id": <slot number>, "input": "<request>"}}."""


def load_config():
    with open("generation-axes.json") as fh:
        axes = json.load(fh)
    with open("generation-topics.json") as fh:
        disciplines = json.load(fh)
    return axes, disciplines


def least_used(values, counts, rng):
    """Pick randomly among the values that have been used the least so far."""
    low = min(counts[v] for v in values)
    return rng.choice([v for v in values if counts[v] == low])


def build_plan(axes, disciplines, per_topic, rng):
    counts = {k: Counter() for k in ["record_type", "complexity", "phrasing", "feature"]}
    record_types = list(axes["record_types"])
    complexities = list(axes["complexity"])
    calls = []
    for di, d in enumerate(disciplines):
        topics = d["topics"]
        half = (len(topics) + 1) // 2
        for hi, chunk in enumerate([topics[:half], topics[half:]]):
            call_id = f"{di:02d}{'ab'[hi]}"
            slots = []
            for topic in chunk:
                used_rt = set()
                used_pairs = set()
                for _ in range(per_topic):
                    if len(used_rt) == len(record_types):
                        used_rt = set()
                    rt = least_used([r for r in record_types if r not in used_rt],
                                    counts["record_type"], rng)
                    cx = least_used([c for c in complexities if (rt, c) not in used_pairs]
                                    or complexities, counts["complexity"], rng)
                    used_rt.add(rt)
                    used_pairs.add((rt, cx))
                    applicable = [f for f, rts in axes["features"].items() if rt in rts]
                    rng.shuffle(applicable)
                    applicable.sort(key=lambda f: counts["feature"][f])
                    features = applicable[: int(cx) - 1]
                    phrasing = least_used(axes["phrasing"], counts["phrasing"], rng)
                    for k, v in [("record_type", rt), ("complexity", cx), ("phrasing", phrasing)]:
                        counts[k][v] += 1
                    for f in features:
                        counts["feature"][f] += 1
                    slots.append({
                        "slot": f"{call_id}-{len(slots) + 1:02d}",
                        "discipline": d["discipline"],
                        "topic": topic,
                        "record_type": rt,
                        "complexity": cx,
                        "features": features,
                        "phrasing": phrasing,
                    })
            calls.append({"call": call_id, "discipline": d["discipline"], "slots": slots})
    return calls


def format_slot(i, s, axes):
    lines = [
        f"{i}. topic: {s['topic']}",
        f"   wants to find: {s['record_type']} ({axes['record_types'][s['record_type']]})",
        f"   complexity: {s['complexity']} - {axes['complexity'][s['complexity']]}",
    ]
    if s["features"]:
        lines.append(f"   constraints: {'; '.join(s['features'])}")
    lines.append(f"   phrasing: {s['phrasing']}")
    return "\n".join(lines)


def generate_call(discipline, slots, axes, thinking):
    """Return {slot id: request} for the slots Gemini answered."""
    prompt = batch_prompt.format(
        discipline=discipline,
        n=len(slots),
        slots="\n\n".join(format_slot(i, s, axes) for i, s in enumerate(slots, 1)),
    )
    cfg = types.GenerateContentConfig(
        temperature=1.0,
        max_output_tokens=30000,
        response_mime_type="application/json",
        response_schema={
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {"id": {"type": "INTEGER"}, "input": {"type": "STRING"}},
                "required": ["id", "input"],
            },
        },
        thinking_config=types.ThinkingConfig(thinking_budget=thinking),
    )
    gemini_usage.check()
    resp = gemini.models.generate_content(
        model=gemini_model, contents=prompt, config=cfg
    )
    gemini_usage.record(resp.usage_metadata, "generate-inputs call")
    out = {}
    for item in json.loads(resp.text):
        i = item.get("id")
        text = (item.get("input") or "").strip()
        if isinstance(i, int) and 1 <= i <= len(slots) and text:
            out[slots[i - 1]["slot"]] = text
    return out


def normalize(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="generated-inputs.jsonl")
    ap.add_argument("--per-topic", type=int, default=8)
    ap.add_argument("--thinking", type=int, default=1024, help="thinking token budget per call")
    ap.add_argument("--retries", type=int, default=2, help="re-asks for slots missing from a call")
    ap.add_argument("--limit-calls", type=int, default=None, help="only run the first N calls")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    axes, disciplines = load_config()
    plan_fn = os.path.splitext(args.out)[0] + ".plan.json"
    if os.path.exists(plan_fn):
        with open(plan_fn) as fh:
            plan = json.load(fh)
        print(f"Loaded plan from {plan_fn}")
    else:
        plan = build_plan(axes, disciplines, args.per_topic, random.Random(args.seed))
        with open(plan_fn, "w") as fh:
            json.dump(plan, fh, indent=2, ensure_ascii=False)
        print(f"Wrote plan to {plan_fn}")

    done = {}
    if os.path.exists(args.out):
        with open(args.out) as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["spec"]["slot"]] = rec["input"]
    seen = {normalize(t) for t in done.values()}

    todo = []
    for call in plan[: args.limit_calls]:
        missing = [s for s in call["slots"] if s["slot"] not in done]
        if missing:
            todo.append((call, missing))
    n_slots = sum(len(c["slots"]) for c in plan)
    print(f"{len(plan)} calls / {n_slots} slots in plan; {len(done)} done; "
          f"{len(todo)} calls to run")

    lock = threading.Lock()
    stats = Counter()

    def run_call(call, missing, out):
        for attempt in range(args.retries + 1):
            got = generate_call(call["discipline"], missing, axes, args.thinking)
            with lock:
                for s in missing:
                    text = got.get(s["slot"])
                    if text is None:
                        continue
                    nt = normalize(text)
                    if nt in seen:
                        stats["duplicates"] += 1
                        continue
                    seen.add(nt)
                    done[s["slot"]] = text
                    out.write(json.dumps({"input": text, "spec": s}, ensure_ascii=False) + "\n")
                    stats["accepted"] += 1
                out.flush()
            missing = [s for s in missing if s["slot"] not in done]
            if not missing:
                break
            print(f"  {call['call']}: {len(missing)} slots missing after attempt {attempt + 1}")
        stats["missing"] += len(missing)
        return call["call"], len(call["slots"]) - len(missing), len(call["slots"])

    with open(args.out, "a") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_call, c, m, out) for c, m in todo]
        try:
            for fut in as_completed(futures):
                try:
                    cid, ok, total = fut.result()
                    print(f"[{len(done)}/{n_slots}] call {cid}: {ok}/{total} slots; "
                          f"spend ${gemini_usage.spent():.4f}")
                except gemini_usage.BudgetExceeded:
                    raise
                except Exception as e:
                    print(f"Call failed: {e!r}")
                    stats["failed_calls"] += 1
        except gemini_usage.BudgetExceeded as e:
            print(f"ABORTED: {e}")
            for fut in futures:
                fut.cancel()

    print(f"Done: {dict(stats)}; {len(done)}/{n_slots} slots filled; "
          f"Gemini spend so far ${gemini_usage.spent():.4f}")


if __name__ == "__main__":
    main()
