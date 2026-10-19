"""
Serve viewer.html and store translation quality ratings in SQLite.

Usage: python viewer-server.py [--port 8000] [--db ratings.sqlite]
then open http://localhost:8000/viewer.html

Ratings are stored per results file (source), so the Gemini and local translations
of the same slot are rated separately.

API:
  GET  /api/sources                         -> ["translated-inputs.jsonl", ...]
  GET  /api/ratings?source=FILE             -> {slot: {"rating": 1-3, "rated_at": ...}}
  POST /api/ratings {"source", "slot", "rating"}
       rating 1 (bad), 2 (acceptable), 3 (good), or null to clear
"""

import argparse
import json
import glob
import os
import re
import sqlite3
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCE = "translated-inputs.jsonl"
SOURCE_RE = re.compile(r"^translated-inputs[\w.-]*\.jsonl$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS ratings (
    source TEXT NOT NULL,  -- results file the rated translation came from
    slot TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating IN (1, 2, 3)),
    input TEXT,
    translation TEXT,  -- JSON snapshot of the options that were rated
    rated_at TEXT NOT NULL,
    PRIMARY KEY (source, slot)
)
"""


def sources():
    return sorted(os.path.basename(f) for f in glob.glob(os.path.join(HERE, "translated-inputs*.jsonl"))
                  if SOURCE_RE.match(os.path.basename(f)))


def valid_source(source):
    return isinstance(source, str) and SOURCE_RE.match(source) and source in sources()


class Store:
    def __init__(self, fn):
        self.db = sqlite3.connect(fn, check_same_thread=False)
        cols = [r[1] for r in self.db.execute("PRAGMA table_info(ratings)")]
        if cols and "source" not in cols:
            # older single-source table: move its ratings to the default source
            self.db.execute("ALTER TABLE ratings RENAME TO ratings_old")
            self.db.execute(SCHEMA)
            self.db.execute(
                "INSERT INTO ratings SELECT ?, slot, rating, input, translation, rated_at "
                "FROM ratings_old", (DEFAULT_SOURCE,))
            self.db.execute("DROP TABLE ratings_old")
        self.db.execute(SCHEMA)
        self.db.commit()
        self.lock = threading.Lock()

    def all(self, source):
        with self.lock:
            rows = self.db.execute(
                "SELECT slot, rating, rated_at FROM ratings WHERE source = ?", (source,)
            ).fetchall()
        return {s: {"rating": r, "rated_at": t} for s, r, t in rows}

    def set(self, source, slot, rating):
        with self.lock:
            if rating is None:
                self.db.execute("DELETE FROM ratings WHERE source = ? AND slot = ?", (source, slot))
            else:
                rec = find_translation(source, slot)
                self.db.execute(
                    "INSERT INTO ratings (source, slot, rating, input, translation, rated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(source, slot) DO UPDATE SET "
                    "rating = excluded.rating, input = excluded.input, "
                    "translation = excluded.translation, rated_at = excluded.rated_at",
                    (source, slot, rating, rec and rec["input"],
                     rec and json.dumps(rec["options"], ensure_ascii=False),
                     time.strftime("%Y-%m-%dT%H:%M:%S")),
                )
            self.db.commit()


def find_translation(source, slot):
    with open(os.path.join(HERE, source)) as fh:
        for line in fh:
            if f'"slot": "{slot}"' in line:
                rec = json.loads(line)
                if rec["slot"] == slot:
                    return rec
    return None


class Handler(SimpleHTTPRequestHandler):
    store = None

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/sources":
            return self.send_json(sources())
        if url.path == "/api/ratings":
            source = parse_qs(url.query).get("source", [DEFAULT_SOURCE])[0]
            if not valid_source(source):
                return self.send_json({"error": "unknown source"}, 400)
            return self.send_json(self.store.all(source))
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/ratings":
            return self.send_json({"error": "not found"}, 404)
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            source = data.get("source", DEFAULT_SOURCE)
            slot, rating = data["slot"], data.get("rating")
            if not valid_source(source):
                raise ValueError("unknown source")
            if not isinstance(slot, str) or rating not in (None, 1, 2, 3):
                raise ValueError("bad slot or rating")
        except Exception as e:
            return self.send_json({"error": str(e)}, 400)
        self.store.set(source, slot, rating)
        self.send_json({"source": source, "slot": slot, "rating": rating})

    def log_message(self, fmt, *args):
        if self.command == "POST":
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--db", default=os.path.join(HERE, "ratings.sqlite"))
    args = ap.parse_args()

    Handler.store = Store(args.db)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=HERE))
    print(f"Viewer at http://localhost:{args.port}/viewer.html (ratings in {args.db})")
    server.serve_forever()


if __name__ == "__main__":
    main()
