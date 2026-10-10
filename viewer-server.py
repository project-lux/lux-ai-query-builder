"""
Serve viewer.html and store translation quality ratings in SQLite.

Usage: python viewer-server.py [--port 8000] [--db ratings.sqlite]
then open http://localhost:8000/viewer.html

API:
  GET  /api/ratings                    -> {slot: {"rating": 1-3, "rated_at": ...}}
  POST /api/ratings {"slot", "rating"}  rating 1 (bad), 2 (acceptable), 3 (good), or null to clear
"""

import argparse
import json
import os
import sqlite3
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSLATED_FN = os.path.join(HERE, "translated-inputs.jsonl")

SCHEMA = """
CREATE TABLE IF NOT EXISTS ratings (
    slot TEXT PRIMARY KEY,
    rating INTEGER NOT NULL CHECK (rating IN (1, 2, 3)),
    input TEXT,
    translation TEXT,  -- JSON snapshot of the options that were rated
    rated_at TEXT NOT NULL
)
"""


class Store:
    def __init__(self, fn):
        self.db = sqlite3.connect(fn, check_same_thread=False)
        self.db.execute(SCHEMA)
        self.db.commit()
        self.lock = threading.Lock()

    def all(self):
        with self.lock:
            rows = self.db.execute("SELECT slot, rating, rated_at FROM ratings").fetchall()
        return {s: {"rating": r, "rated_at": t} for s, r, t in rows}

    def set(self, slot, rating):
        with self.lock:
            if rating is None:
                self.db.execute("DELETE FROM ratings WHERE slot = ?", (slot,))
            else:
                rec = find_translation(slot)
                self.db.execute(
                    "INSERT INTO ratings (slot, rating, input, translation, rated_at) "
                    "VALUES (?, ?, ?, ?, ?) ON CONFLICT(slot) DO UPDATE SET "
                    "rating = excluded.rating, input = excluded.input, "
                    "translation = excluded.translation, rated_at = excluded.rated_at",
                    (slot, rating, rec and rec["input"],
                     rec and json.dumps(rec["options"], ensure_ascii=False),
                     time.strftime("%Y-%m-%dT%H:%M:%S")),
                )
            self.db.commit()


def find_translation(slot):
    with open(TRANSLATED_FN) as fh:
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
        if self.path.split("?")[0] == "/api/ratings":
            return self.send_json(self.store.all())
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/ratings":
            return self.send_json({"error": "not found"}, 404)
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            slot, rating = data["slot"], data.get("rating")
            if not isinstance(slot, str) or rating not in (None, 1, 2, 3):
                raise ValueError("bad slot or rating")
        except Exception as e:
            return self.send_json({"error": str(e)}, 400)
        self.store.set(slot, rating)
        self.send_json({"slot": slot, "rating": rating})

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
