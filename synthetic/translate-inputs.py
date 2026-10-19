"""
Translate generated requests (generated-inputs.jsonl) into LUX queries via the
query-cli.py pipeline, and optionally fill in hit counts from the live LUX API.

Results are appended to translated-inputs.jsonl keyed by slot id as soon as each one
finishes. Slots already in the output are never translated again, so an aborted run
(budget, crash, Ctrl-C) can simply be re-run to continue.

Slots are translated in a fixed, balanced order (round robin across discipline x
complexity), so --count 300 then --count 600 extends the same sample.

Usage:
  python translate-inputs.py --count 300     # ensure 300 slots are translated
  python translate-inputs.py --hits          # fill in missing hit counts, no Gemini calls
  python translate-inputs.py --backend local --count 4800
      # use the local MTPLX server instead of Gemini; writes translated-inputs-local.jsonl

Don't run a translation and a --hits pass at the same time: --hits rewrites the file.
"""

import argparse
import importlib.util
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import requests

import gemini_usage

spec = importlib.util.spec_from_file_location("test_queries", "test-queries.py")
tq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tq)


def load_jsonl(fn):
    if not os.path.exists(fn):
        return []
    with open(fn) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def balanced_order(records, seed):
    """Order slots so any prefix is balanced across disciplines and complexities.

    Strata are visited so that consecutive picks cycle through every complexity and
    every discipline, and each (discipline, complexity) pair once per pass."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for r in records:
        strata[(r["spec"]["discipline"], r["spec"]["complexity"])].append(r)
    disciplines = sorted({d for d, _ in strata})
    complexities = sorted({c for _, c in strata})
    rng.shuffle(disciplines)
    for k in strata:
        rng.shuffle(strata[k])
    nd, nc = len(disciplines), len(complexities)
    keys = [(disciplines[i % nd], complexities[(i // nd + i) % nc]) for i in range(nd * nc)]
    order = []
    while any(strata.values()):
        for k in keys:
            if strata[k]:
                order.append(strata[k].pop())
    return order


def translate(args):
    inputs = load_jsonl(args.inputs)
    existing = load_jsonl(args.out)
    done = {r["slot"] for r in existing if not (args.retry_errors and r["error"])}
    if args.retry_errors:
        # rewrite without the errored records so they are replaced, not duplicated
        keep = [r for r in existing if not r["error"]]
        if len(keep) != len(existing):
            write_jsonl(args.out, keep)

    order = balanced_order(inputs, args.seed)[: args.count]
    todo = [r for r in order if r["spec"]["slot"] not in done]
    print(f"{len(inputs)} inputs; {len(done)} already translated; "
          f"{len(todo)} to translate for --count {args.count}; "
          f"Gemini spend so far ${gemini_usage.spent():.4f}")

    if args.backend == "local":
        tq.qc.local_thinking = not args.no_think
        generate = tq.qc.generate_local
        model = tq.qc.local_model + ("" if tq.qc.local_thinking else " (no thinking)")
    else:
        generate = tq.qc.generate_gemini
        model = tq.qc.gemini_model

    stats = Counter()
    with open(args.out, "a") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(tq.run_one, r["input"], generate=generate): r for r in todo}
        try:
            for fut in as_completed(futures):
                rec = futures[fut]
                try:
                    res = fut.result()
                except (gemini_usage.BudgetExceeded, requests.ConnectionError):
                    raise
                except Exception as e:
                    print(f"Failed {rec['spec']['slot']}: {e!r}")
                    stats["failed"] += 1
                    continue
                row = {"slot": rec["spec"]["slot"], "input": rec["input"],
                       "spec": rec["spec"], "options": res["options"], "error": res["error"],
                       "model": model}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                stats["errors" if res["error"] else "translated"] += 1
                n = sum(stats.values())
                print(f"[{n}/{len(todo)}] {row['slot']}: "
                      f"{sum(1 for o in res['options'] if not o.get('keyword'))} non-keyword queries"
                      + (f" ERROR {res['error']}" if res["error"] else ""))
                sys.stdout.flush()
        except (gemini_usage.BudgetExceeded, requests.ConnectionError) as e:
            print(f"ABORTED: {e}")
            for f in futures:
                f.cancel()
    print(f"Done: {dict(stats)}; Gemini spend so far ${gemini_usage.spent():.4f}")


def write_jsonl(fn, rows):
    tmp = fn + ".tmp"
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, fn)


def fill_hits(args):
    rows = load_jsonl(args.out)
    todo = [o for r in rows for o in r["options"]
            if not o.get("keyword") and "lux_query" in o
            and (o.get("hits") is None or args.recheck)]
    print(f"Fetching hit counts for {len(todo)} queries")
    tq.check_api()

    def fetch(o):
        o["hits"], err = tq.hit_count(o["scope"], o["lux_query"])
        if err:
            o["hits_error"] = err
        else:
            o.pop("hits_error", None)

    with ThreadPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(as_completed([pool.submit(fetch, o) for o in todo]), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}")
    write_jsonl(args.out, rows)
    print(f"Updated {args.out}; {sum(1 for o in todo if o.get('hits'))} queries with hits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="generated-inputs.jsonl")
    ap.add_argument("--backend", choices=["gemini", "local"], default="gemini")
    ap.add_argument("--no-think", action="store_true", help="local backend: disable model thinking")
    ap.add_argument("--out", default=None,
                    help="default translated-inputs.jsonl, or translated-inputs-local.jsonl for --backend local")
    ap.add_argument("--count", type=int, default=0,
                    help="total number of slots (in balanced order) that should be translated")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-translate slots whose earlier translation failed")
    ap.add_argument("--hits", action="store_true",
                    help="fill in missing hit counts from LUX instead of translating")
    ap.add_argument("--recheck", action="store_true", help="with --hits, refetch all hit counts")
    ap.add_argument("--workers", type=int, default=None, help="default 8 for gemini, 2 for local")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.out is None:
        args.out = "translated-inputs-local.jsonl" if args.backend == "local" else "translated-inputs.jsonl"
    if args.workers is None:
        args.workers = 2 if args.backend == "local" else 8

    if args.hits:
        fill_hits(args)
    else:
        translate(args)


if __name__ == "__main__":
    main()
