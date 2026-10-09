"""
Generate a large, varied set of natural language search requests for testing the query
processor. Each batch is seeded by a sampled combination of topic, record type, features,
complexity, phrasing and persona (generation-axes.json + generation-topics.json), so
previous output never needs to be fed back to the model. Near duplicates are removed
locally with embeddings. Output is appended to a JSONL file, so runs can be resumed or
extended.

Usage: python generate-inputs.py --target 1000 [--batch-size 10] [--out generated-inputs.jsonl]
"""

import argparse
import json
import os
import random
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from google import genai
from google.genai import types

import gemini_usage

with open("google_project.txt") as fh:
    gcp_name = fh.read().strip()

gemini_model = "gemini-3.8-flash"
embedding_model = "text-embedding-005"
gemini = genai.Client(vertexai=True, project=gcp_name, location="global")

batch_prompt = """LUX is Yale University's unified collections discovery platform, covering Yale's
libraries, museums and archives: books, journals, manuscripts, artworks, photographs,
maps, natural history specimens, archival collections, people, organizations, places,
concepts and events.

Imagine {n} different Yale students, each describing to a helpful librarian what they
want to find in LUX. Write down what each one says.

All of them:
  * are a {persona}
  * are interested in the topic: {topic} (within {discipline})
  * want to find {record_type} ({record_desc})
  * phrase their request in this style: {phrasing}
  * make a request with complexity: {complexity_desc}
{feature_text}
Each request must be a natural language DESCRIPTION of what the student wants, as they
would say it out loud, not a list of search keywords. For example:
  BAD (keywords):  "medieval maps"
  GOOD (natural):  "I'm looking for maps that were made during the Middle Ages"

Make the {n} requests clearly different from each other: approach the topic from
different angles and pick different specific aspects, people, places, periods or examples.
Keep every request realistic: something a real student would plausibly want and could
reasonably expect a university collection to have. Even detailed requests should not
pile up obscure, improbable details. Use ordinary everyday language. Do not use database field names, query syntax, boolean
operators, or data model terminology.

Return a JSON array of {n} strings."""


def load_config():
    with open("generation-axes.json") as fh:
        axes = json.load(fh)
    with open("generation-topics.json") as fh:
        topics = [
            (d["discipline"], t) for d in json.load(fh) for t in d["topics"]
        ]
    return axes, topics


def least_used(values, counts, rng):
    """Pick randomly among the values that have been used the least so far."""
    low = min(counts[v] for v in values)
    return rng.choice([v for v in values if counts[v] == low])


def sample_spec(axes, topics, counts, rng):
    discipline, topic = least_used(topics, counts["topic"], rng)
    record_type = least_used(list(axes["record_types"]), counts["record_type"], rng)
    complexity = least_used(list(axes["complexity"]), counts["complexity"], rng)
    phrasing = least_used(axes["phrasing"], counts["phrasing"], rng)
    persona = least_used(axes["personas"], counts["persona"], rng)

    applicable = [f for f, rts in axes["features"].items() if record_type in rts]
    n_features = min(int(complexity) - 1, len(applicable))
    rng.shuffle(applicable)
    applicable.sort(key=lambda f: counts["feature"][f])
    features = applicable[:n_features]

    return {
        "discipline": discipline,
        "topic": topic,
        "record_type": record_type,
        "complexity": complexity,
        "phrasing": phrasing,
        "persona": persona,
        "features": features,
    }


def count_spec(counts, spec):
    counts["topic"][(spec["discipline"], spec["topic"])] += 1
    for k in ["record_type", "complexity", "phrasing", "persona"]:
        counts[k][spec[k]] += 1
    for f in spec["features"]:
        counts["feature"][f] += 1


def generate_batch(spec, axes, n):
    if spec["features"]:
        feature_text = "  * include constraints on: " + "; ".join(spec["features"]) + "\n"
    else:
        feature_text = ""
    prompt = batch_prompt.format(
        n=n,
        persona=spec["persona"],
        topic=spec["topic"],
        discipline=spec["discipline"],
        record_type=spec["record_type"],
        record_desc=axes["record_types"][spec["record_type"]],
        phrasing=spec["phrasing"],
        complexity_desc=axes["complexity"][spec["complexity"]],
        feature_text=feature_text,
    )
    cfg = types.GenerateContentConfig(
        temperature=1.0,
        response_mime_type="application/json",
        response_schema={"type": "ARRAY", "items": {"type": "STRING"}},
    )
    gemini_usage.check()
    resp = gemini.models.generate_content(
        model=gemini_model, contents=prompt, config=cfg
    )
    gemini_usage.record(resp.usage_metadata, "generate-inputs batch")
    return [s.strip() for s in json.loads(resp.text) if s.strip()]


def normalize(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def embed(texts):
    vecs = []
    for i in range(0, len(texts), 100):
        gemini_usage.check()
        resp = gemini.models.embed_content(
            model=embedding_model, contents=texts[i : i + 100]
        )
        gemini_usage.record_embedding(resp, "generate-inputs embedding")
        vecs.extend(e.values for e in resp.embeddings)
    arr = np.array(vecs, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


class Deduper:
    def __init__(self, threshold, cache_fn):
        self.threshold = threshold
        self.cache_fn = cache_fn
        self.seen = set()
        self.vecs = np.zeros((0, 0), dtype=np.float32)

    def load(self, texts):
        self.seen = {normalize(t) for t in texts}
        if not texts:
            return
        if os.path.exists(self.cache_fn):
            vecs = np.load(self.cache_fn)
            if len(vecs) == len(texts):
                self.vecs = vecs
                return
        print(f"Embedding {len(texts)} existing inputs...")
        self.vecs = embed(texts)
        np.save(self.cache_fn, self.vecs)

    def filter(self, texts):
        """Return (accepted, rejected) and record the accepted ones."""
        cands = []
        rejected = []
        for t in texts:
            nt = normalize(t)
            if nt in self.seen:
                rejected.append(t)
            else:
                self.seen.add(nt)
                cands.append(t)
        if not cands:
            return [], rejected
        new = embed(cands)
        accepted = []
        keep = []
        for t, v in zip(cands, new):
            sims = [self.vecs @ v] if len(self.vecs) else []
            if keep:
                sims.append(np.array(keep) @ v)
            if any(len(s) and s.max() >= self.threshold for s in sims):
                rejected.append(t)
            else:
                accepted.append(t)
                keep.append(v)
        if keep:
            k = np.array(keep)
            self.vecs = np.vstack([self.vecs, k]) if len(self.vecs) else k
            np.save(self.cache_fn, self.vecs)
        return accepted, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1000, help="total inputs wanted in --out")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--out", default="generated-inputs.jsonl")
    ap.add_argument("--threshold", type=float, default=0.92, help="cosine similarity cutoff")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    axes, topics = load_config()
    rng = random.Random(args.seed)
    counts = {k: Counter() for k in
              ["topic", "record_type", "complexity", "phrasing", "persona", "feature"]}

    existing = []
    if os.path.exists(args.out):
        with open(args.out) as fh:
            existing = [json.loads(line) for line in fh if line.strip()]
    for rec in existing:
        count_spec(counts, rec["spec"])
    deduper = Deduper(args.threshold, os.path.splitext(args.out)[0] + ".embeddings.npy")
    deduper.load([r["input"] for r in existing])

    total = len(existing)
    print(f"{total} existing inputs, target {args.target}")
    lock = threading.Lock()
    stats = Counter()

    def make_spec():
        with lock:
            spec = sample_spec(axes, topics, counts, rng)
            count_spec(counts, spec)
            return spec

    with open(args.out, "a") as out, ThreadPoolExecutor(args.workers) as pool:
        pending = set()
        try:
            while total < args.target or pending:
                while total + len(pending) * args.batch_size < args.target and \
                        len(pending) < args.workers * 2:
                    spec = make_spec()
                    fut = pool.submit(generate_batch, spec, axes, args.batch_size)
                    fut.spec = spec
                    pending.add(fut)
                if not pending:
                    break
                done = next(as_completed(pending))
                pending.remove(done)
                try:
                    texts = done.result()
                except gemini_usage.BudgetExceeded:
                    raise
                except Exception as e:
                    print(f"Batch failed ({done.spec['topic']}): {e}")
                    stats["failed_batches"] += 1
                    continue
                accepted, rejected = deduper.filter(texts)
                for t in accepted:
                    out.write(json.dumps({"input": t, "spec": done.spec}, ensure_ascii=False) + "\n")
                out.flush()
                total += len(accepted)
                stats["accepted"] += len(accepted)
                stats["rejected"] += len(rejected)
                print(f"[{total}/{args.target}] +{len(accepted)} -{len(rejected)} "
                      f"{done.spec['topic']!r} / {done.spec['record_type']}")
        except gemini_usage.BudgetExceeded as e:
            print(f"ABORTED: {e}")
            for fut in pending:
                fut.cancel()

    print(f"Done: {dict(stats)}; Gemini spend so far ${gemini_usage.spent():.4f}")


if __name__ == "__main__":
    main()
