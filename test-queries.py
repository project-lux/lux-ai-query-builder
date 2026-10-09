"""
Generate realistic student search queries with Gemini, run them through the
query-cli.py processing pipeline, and record hit counts from the live LUX API.

Usage: python test-queries.py [--count 100] [--queries FILE] [--out FILE]
"""

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus

import requests
from google.genai import types

import gemini_usage

spec = importlib.util.spec_from_file_location("query_cli", "query-cli.py")
qc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qc)

generator_model = "gemini-3.8-flash"
LUX_API = "https://lux.collections.yale.edu/api"
# Disabled while the live API returns 0 for relationship searches; enable with --hits
CHECK_HITS = False

generator_prompt = """You are helping test the search interface of LUX, Yale University's
unified collections discovery platform. LUX covers Yale's libraries, the Yale University
Art Gallery, the Yale Center for British Art, the Peabody Museum of Natural History, the
Beinecke Rare Book & Manuscript Library, archives, and more: books, manuscripts, artworks,
photographs, maps, specimens, archival collections, people, organizations, places,
concepts and events.

Imagine {count} different Yale university students, each describing to a helpful
librarian what they are looking for. Write down what each one says.

IMPORTANT: each entry must be a natural language DESCRIPTION of what the student wants,
written as a full sentence or request, as they would say it out loud. It must NOT be a
list of search keywords. For example:
  BAD (keywords):  "medieval maps"
  GOOD (natural):  "I'm looking for maps that were made during the Middle Ages"
  BAD (keywords):  "turner watercolors venice"
  GOOD (natural):  "Can you show me watercolors J.M.W. Turner painted of Venice?"

Make them varied:
  * some broad and general, some narrow and very specific
  * some very simple, some long and complicated with several constraints (dates,
    places, materials, creators, subjects, formats, languages, etc.)
  * cover many disciplines: art history, history, literature, natural science,
    anthropology, music, architecture, religion, politics, Yale history, and so on
  * ask for objects, books, archives, people, places, organizations and events
  * vary the phrasing: questions, polite requests, casual descriptions, statements
    of a research need
  * a few may contain realistic typos or vague wording

Use ordinary everyday language. Do NOT use database field names, query syntax,
boolean operators, or data model terminology.

Return a JSON array of {count} strings and nothing else."""


def generate_queries(count):
    cfg = types.GenerateContentConfig(
        temperature=1.0,
        response_mime_type="application/json",
        response_schema={"type": "ARRAY", "items": {"type": "STRING"}},
    )
    gemini_usage.check()
    resp = qc.gemini.models.generate_content(
        model=generator_model,
        contents=generator_prompt.format(count=count),
        config=cfg,
    )
    gemini_usage.record(resp.usage_metadata, "test-queries generate")
    queries = json.loads(resp.text)
    # de-duplicate, keep order
    seen = set()
    return [q for q in queries if not (q in seen or seen.add(q))][:count]


def is_keyword(opt):
    if opt.get("scope") == "ALL":
        return True
    lq = opt.get("lux_query", {})
    return "text" in lq or (
        "AND" in lq and all(list(x.keys())[0] == "text" for x in lq["AND"])
    )


def hit_count(scope, lq):
    qstr = quote_plus(json.dumps(lq, separators=(",", ":")))
    url = f"{LUX_API}/search-estimate/{scope}?q={qstr}"
    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}: {resp.text[:300]}"
        return resp.json().get("totalItems"), None
    except Exception as e:
        return None, str(e)


def run_one(user_input, attempts=2):
    result = {"input": user_input, "options": [], "error": None}
    js = None
    for _ in range(attempts):
        try:
            js = qc.generate_gemini(user_input)
        except gemini_usage.BudgetExceeded:
            raise
        except Exception as e:
            result["error"] = f"gemini: {e}"
            js = None
        if js:
            break
    if not js:
        result["error"] = result["error"] or "no valid JSON from model"
        return result
    result["error"] = None
    if isinstance(js, list):
        js = {"options": js}

    for opt in js.get("options", []):
        out = {
            "natural": opt.get("natural"),
            "scope": opt.get("scope"),
            "model_query": opt.get("query"),
        }
        try:
            scope = opt["scope"]
            lq = qc.post_process(opt["query"], None if scope == "ALL" else scope)
            out["lux_query"] = lq
        except Exception as e:
            out["error"] = f"post_process: {e!r}"
            result["options"].append(out)
            continue
        out["keyword"] = is_keyword(out)
        if not out["keyword"]:
            if CHECK_HITS:
                out["hits"], err = hit_count(scope, lq)
                if err:
                    out["error"] = err
            qstr = quote_plus(json.dumps(lq, separators=(",", ":")))
            out["url"] = f"{qc.LUX_HOST}/view/results/{qc.uri_scopes[scope]}?q={qstr}"
        result["options"].append(out)
    return result


def check_api():
    # Relationship searches have been seen returning 0 for everything on the live API
    lq = {"classification": {"name": "painting"}}
    hits, err = hit_count("item", lq)
    if not hits:
        print(f"WARNING: sanity query {lq} returned {hits} {err or ''}; "
              "hit counts for nested queries are probably unreliable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--queries", default="test-queries-input.json")
    ap.add_argument("--out", default="test-queries-results.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hits", action="store_true", help="fetch hit counts from LUX")
    args = ap.parse_args()

    global CHECK_HITS
    CHECK_HITS = CHECK_HITS or args.hits

    if CHECK_HITS:
        check_api()
    if os.path.exists(args.queries):
        with open(args.queries) as fh:
            queries = json.load(fh)
        print(f"Loaded {len(queries)} queries from {args.queries}")
    else:
        queries = generate_queries(args.count)
        with open(args.queries, "w") as fh:
            json.dump(queries, fh, indent=2)
        print(f"Generated {len(queries)} queries -> {args.queries}")

    results = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, res in enumerate(pool.map(run_one, queries), 1):
                results.append(res)
                hits = [o.get("hits") for o in res["options"] if not o.get("keyword")]
                summary = f"hits {hits}" if CHECK_HITS else f"{len(hits)} non-keyword queries"
                print(f"[{i}/{len(queries)}] {res['input']!r} -> {summary}"
                      + (f" ERROR {res['error']}" if res["error"] else ""))
                sys.stdout.flush()
    except gemini_usage.BudgetExceeded as e:
        print(f"ABORTED: {e}")
    finally:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"Wrote {len(results)} results to {args.out}; "
              f"Gemini spend so far ${gemini_usage.spent():.4f}")


if __name__ == "__main__":
    main()
