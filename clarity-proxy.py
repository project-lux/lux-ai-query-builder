import sys
from flask import Flask, request
import json
import requests
from urllib.parse import quote_plus


import weaviate
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.classes.init import Auth

from sentence_transformers import SentenceTransformer

from clarity import Clarity

import vertexai.generative_models as generative_models
from vertexai.generative_models import GenerativeModel, Part
from google import genai
from google.genai import types

from jsonschema import Draft202012Validator

schemafn = "generated_schema.json"
fh = open(schemafn)
schema = json.load(fh)
fh.close()
# validator = Draft202012Validator(schema)
validator = None

###
### NOTE WELL
###
### Before production, UI needs to send a session token
### Then we need to generate a new session per react session
### Otherwise, "I want my previous query" will return the previous
### user's query in the same session
### OR... just turn off all memory (current solution)

# GPT4o
with open("../clarity-config.json") as fh:
    config = json.load(fh)

# Claude 3.5
with open("../clarity-claude-config.json") as fh:
    config2 = json.load(fh)

WEAVIATE_URL = config2.get("weaviate_url", "")
WEAVIATE_KEY = config2.get("weaviate_key", "")

with open("google_project.txt") as fh:
    gcp_name = fh.read().strip()

LUX_HOST = "https://lux.collections.yale.edu"

client = Clarity(
    base_url=config["base_url"],
    instance_id=config["instance_id"],
    api_key=config["private_access_key"],
    agent_name=config["agent_name"],
)
session = client.create_session("proxy-test")
client2 = Clarity(
    base_url=config2["base_url"],
    instance_id=config2["instance_id"],
    api_key=config2["private_access_key"],
    agent_name=config2["agent_name"],
)
session2 = client2.create_session("proxy-test2")

if not WEAVIATE_KEY:
    weave = weaviate.connect_to_custom(url=WEAVIATE_URL)
else:
    weave = weaviate.connect_to_weaviate_cloud(cluster_url=WEAVIATE_URL, auth_credentials=Auth.api_key(WEAVIATE_KEY))
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

with open("system-prompt-merged.txt") as fh:
    textsi_1 = fh.read().strip()

gemini_model = "gemini-2.5-flash-preview-04-17"
# gemini = "gemini-2.0-flash-001"
generated_config = types.GenerateContentConfig(
    temperature=0.8,
    top_p=0.95,
    max_output_tokens=8192,
    response_modalities=["TEXT"],
    safety_settings=[
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ],
    response_mime_type="application/json",
    thinking_config=types.ThinkingConfig(thinking_budget=0),
    system_instruction=[types.Part.from_text(text=textsi_1)],
)
gemini = genai.Client(
    vertexai=True,
    project=gcp_name,
    location="us-central1",
)

scopes = {}
all_scopes = ["place", "event", "set", "item", "work", "agent", "concept"]
r = requests.get(f"{LUX_HOST}/api/advanced-search-config")
asc = r.json()["terms"]
for sc, tms in asc.items():
    scopes[sc] = {}
    for k, v in tms.items():
        if v["relation"] in all_scopes:
            scopes[sc][k] = v["relation"]

scope_class = {
    "place": ["place"],
    "agent": ["person", "group"],
    "concept": ["type"],
    "event": ["event"],
    "item": ["HumanMadeObject"],
    "work": ["LinguisticObject"],
    "set": ["Set"],
}

##### Functions


def generate(client, prompt):
    if client == "gemini":
        return generate_gemini(prompt)
    resp = client.complete(prompt, parse_json=True)
    try:
        return resp["json"]
    except Exception as e:
        return None


def generate_gemini(prompt):
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    output = []
    cfg = generated_config
    for chunk in gemini.models.generate_content_stream(
        model=gemini_model,
        contents=contents,
        config=cfg,
    ):
        output.append(chunk.text)
    jstr = "".join(output)
    try:
        js = json.loads(jstr)
        return js
    except Exception as e:
        print(jstr)
        sys.stdout.flush()
        return None


def lux_search(scope, query):
    qstr = json.dumps(query)
    encq = quote_plus(qstr)
    url = f"{LUX_HOST}/api/search/{scope}?q={encq}"
    try:
        resp = requests.get(url)
        js = resp.json()

        if "orderedItems" in js:
            return [x["id"] for x in js["orderedItems"]]
        else:
            return []
    except Exception as e:
        print("fetch records broke...")
        print(e)
        return []


def vector_search(query, types=None, limit=5):
    query_vector = embed_model.encode(query).tolist()
    collection = weave.collections.get("WikidataArticle")

    # Create filter for classes if specified
    if types:
        filters = Filter.by_property("classes").contains_any(types)
    else:
        filters = None
    results = collection.query.near_vector(
        near_vector=query_vector,
        filters=filters,
        limit=limit,
        return_metadata=MetadataQuery(distance=True),
        return_properties=["wd_id", "wp_title", "classes"],
    )
    return results.objects


def post_process(query, scope=None):
    new = {}
    if "p" in query:
        # BOOL
        new[query["f"]] = [post_process(x, scope) for x in query["p"]]
    elif "r" in query:
        # Change scope
        scope = scopes[scope].get(query["f"], scope)
        new[query["f"]] = post_process(query["r"], scope)
    else:
        if "d" in query:
            print(f"SAW 'd' in {query}")
            # This is where we reach out to the vector DB
            results = vector_search(query["v"], types=scope_class.get(scope, []))
            # for now, turn it into an identifier search

            ids = []
            for r in results:
                # replace with LUX ids
                print(f"{r.properties['wp_title']} = {r.metadata.distance}")
                if not ids or r.metadata.distance < 0.4:
                    wd_id = r.properties["wd_id"]
                    rids = lux_search(scope, {"identifier": f"http://www.wikidata.org/entity/{wd_id}"})
                    ids.extend(rids)

            new["OR"] = [{"id": x} for x in ids]
            return new

        if query["f"] in ["height", "width", "depth", "dimension"]:
            query["v"] = float(query["v"])
        elif query["f"] in ["hasDigitalImage"]:
            query["v"] = int(query["v"])
        new[query["f"]] = query["v"]
        if "c" in query:
            new["_comp"] = query["c"]
    return new


def test_hits(scope, query):
    encq = quote_plus(query)
    url = f"{LUX_HOST}/api/search-estimate/{scope}?q={encq}"
    try:
        resp = requests.get(url)
        js = resp.json()
        if js["totalItems"] >= 1:
            return True
        else:
            print(f"No hits in {query}\n{js}")
            return False
    except Exception as e:
        print(e)
        return False


def process_query(js):
    scope = js["scope"]
    try:
        lux_q = post_process(js["query"], scope)
    except Exception as e:
        raise e
        return "The javascript generated does not follow the schema laid out. Please try again to find a different structure for the same query."

    if validator is not None:
        try:
            js3 = {"q": lux_q, "scope": scope}
            errs = list(validator.iter_errors(js3))
            if errs:
                err_msg = []
                for e in errs:
                    print(f"  /{'/'.join([str(x) for x in e.absolute_path])} --> {e.message} ")
                    err_msg.append(f"{e.message} in /{'/'.join([str(x) for x in e.absolute_path])}")
                errmsg = "\n".join(err_msg)
                txt = f"The query generated from the javascript returned did not match the final query structure.\
    The messages generated from testing the schema were:\
    {errmsg}\
    Please try again to find a different structure for the same query."
                print(txt)
            lux_q["_scope"] = scope
        except Exception as e:
            return "The javascript generated was not valid. Please try again."
    js["query"] = lux_q
    return js


def build_query_multi(client, q):
    print(q)
    js = generate(client, q)
    if js is None:
        # just try again?
        js = generate(client, q)
        if js is None:
            return "Could not get JSON back from the AI for that query"
    print(js)
    if "options" in js:
        # full set of options
        for o in js["options"]:
            js2 = o["q"]
            js3 = process_query(js2)
            if type(js3) == str:
                # uhoh!
                # trash it?
                return js3
            # pull it up a level
            scope = o["q"]["scope"]
            o["q"] = o["q"]["query"]
            o["q"]["_scope"] = scope
    # We're good
    if len(query_cache) > 128:
        query_cache.popitem()
    query_cache[q] = js
    return js


##### FLASK from here on down


# Refactor to use @functools.lru_cache
query_cache = {}
failed_query_cache = {}

app = Flask(__name__)


@app.after_request
def add_cors(response):
    response.headers["content-type"] = "application/json"
    response.headers["access-control-allow-origin"] = "*"
    return response


@app.route("/api/translate_multi/<string:scope>", methods=["GET"])
def make_query_multi(scope):
    q = request.args.get("q", None)
    if not q:
        return ""
    elif q in query_cache:
        return query_cache[q]

    if q.startswith("!"):
        cl = "gemini"
        q = q[1:]
    elif q.startswith("@"):
        cl = client
        q = q[1:]
    else:
        cl = client2

    js = build_query_multi(cl, q)
    if type(js) == str:
        js = build_query_multi(cl, js + " " + q)
        if type(js) == str:
            failed_query_cache[q] = js
            return json.dumps({"_scope": "item", "name": f"ERROR for query: {js}"})
    jstr = json.dumps(js)
    print(f"Okay: {jstr}")
    return jstr


@app.route("/api/translate_raw/<string:scope>", methods=["GET"])
def make_query_raw(scope):
    q = request.args.get("q", None)
    if not q:
        return ""
    if q.startswith("!"):
        cl = "gemini"
        q = q[1:]
    elif q.startswith("@"):
        cl = client
        q = q[1:]
    else:
        cl = client2
    js = generate(cl, q)
    return json.dumps(js)


@app.route("/api/dump_cache", methods=["GET"])
def dump_cache():
    which = request.args.get("type", "okay")
    if which.startswith("fail"):
        return json.dumps(failed_query_cache)
    else:
        return json.dumps(query_cache)


if __name__ == "__main__":
    if "i" not in sys.argv:
        app.run(debug=True, port=8443, host="0.0.0.0", ssl_context="adhoc")
