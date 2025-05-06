

from flask import Flask, request, redirect, make_response
import json
import requests
from urllib.parse import quote_plus
import copy


import weaviate
from weaviate.classes.config import Property, DataType, Configure, VectorDistances
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.classes.init import Auth

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from clarity import Clarity
from jsonschema import Draft202012Validator


###
### NOTE WELL
###
### Before production, UI needs to send a session token
### Then we need to generate a new session per react session
### Otherwise, "I want my previous query" will return the previous
### user's query in the same session

### OR... just turn off all memory



schemafn = "generated_schema.json"
fh = open(schemafn)
schema = json.load(fh)
fh.close()
validator = Draft202012Validator(schema)

with open('../clarity-config.json') as fh:
    config = json.load(fh)

with open('../clarity-claude-dev-config.json') as fh:
    config2 = json.load(fh)

with open('../clarity-dev-config.json') as fh:
    config3 = json.load(fh)

WEAVIATE_URL = config2.get('weaviate_url', '')
WEAVIATE_KEY = config2.get('weaviate_key', '')

LUX_HOST = "https://lux.collections.yale.edu"

client = Clarity(base_url=config['base_url'], instance_id=config['instance_id'],
        api_key=config['private_access_key'], agent_name=config['agent_name'])
session = client.create_session("proxy-test")
client2 = Clarity(base_url=config2['base_url'], instance_id=config2['instance_id'],
        api_key=config2['private_access_key'], agent_name=config2['agent_name'])
session2 = client2.create_session("proxy-test2")

client3 = Clarity(base_url=config3['base_url'], instance_id=config3['instance_id'],
        api_key=config3['private_access_key'], agent_name=config3['agent_name'])
session3 = client3.create_session("proxy-test3")


if not WEAVIATE_KEY:
    weave = weaviate.connect_to_custom(url=WEAVIATE_URL)
else:
    weave = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_KEY))
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

scopes = {}
all_scopes = ['place', 'event', 'set', 'item', 'work', 'agent', 'concept']
r = requests.get(f"{LUX_HOST}/api/advanced-search-config")
asc = r.json()['terms']
for (sc,tms) in asc.items():
    scopes[sc] = {}
    for (k,v) in tms.items():
        if v['relation'] in all_scopes:
            scopes[sc][k] = v['relation']

scope_class = {
    'place': ['place'],
    'agent': ['person', 'group'],
    'concept': ['type'],
    'event': ['event'],
    'item': ['HumanMadeObject'],
    'work': ['LinguisticObject'],
    'set': ['Set']
}

##### Functions

def generate(client, prompt):
    resp = client.complete(prompt, parse_json=True)
    try:
        return resp['json']
    except:
        return None

def lux_search(scope, query):
    qstr = json.dumps(query)
    encq = quote_plus(qstr)
    url = f"{LUX_HOST}/api/search/{scope}?q={encq}"
    try:
        resp = requests.get(url)
        js = resp.json()
        recs = []
        if 'orderedItems' in js:
            return [x['id'] for x in js['orderedItems']]
        else:
            return []
    except Exception as e:
        print("fetch records broke...")
        print(e)
        return []

def vector_search(query, types=None, limit=6):

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
        return_properties=["wd_id", "wp_title", "wp_text", "classes"]
    )

    return results.objects

def post_process(query, scope=None):
    new = {}
    if 'p' in query:
        # BOOL
        new[query['f']] = [post_process(x, scope) for x in query['p']]
    elif 'r' in query:
        # Change scope
        scope = scopes[scope].get(query['f'], scope)
        new[query['f']] = post_process(query['r'], scope)
    else:
        if 'd' in query:
            print(f"SAW 'd' in {query}")
            # This is where we reach out to the vector DB
            results = vector_search(query['v'], types=scope_class.get(scope, []))
            # for now, turn it into an identifier search

            # replace with LUX ids
            wd_id = results[0].properties['wd_id']
            ids = lux_search(scope, {"identifier": f"http://www.wikidata.org/entity/{wd_id}"})
            new['id'] = ids[0]
            return new

        if query['f'] in ['height', 'width', 'depth', 'dimension']:
            query['v'] = float(query['v'])
        elif query['f'] in ['hasDigitalImage']:
            query['v'] = int(query['v'])
        new[query['f']] = query['v']
        if 'c' in query:
            new['_comp'] = query['c']
    return new


def test_hits(scope, query):
    encq = quote_plus(query)
    url = f"{LUX_HOST}/api/search-estimate/{scope}?q={encq}"
    try:
        resp = requests.get(url)
        js = resp.json()
        if js['totalItems'] >= 1:
            return True
        else:
            print(f"No hits in {query}\n{js}")
            return False
    except Exception as e:
        print(e)
        return False

def process_query(js):
    scope = js['scope']
    try:
        lux_q = post_process(js['query'], scope)
    except:
        return "The javascript generated does not follow the schema laid out. Please try again to find a different structure for the same query."

    try:
        js3 = {"q": lux_q, "scope": scope}
        errs = list(validator.iter_errors(js3))
        if errs:
            err_msg = []
            for e in errs:
                print(f"  /{'/'.join([str(x) for x in e.absolute_path])} --> {e.message} ")
                err_msg.append(f"{e.message} in /{'/'.join([str(x) for x in e.absolute_path])}")
            errmsg = '\n'.join(err_msg)
            txt = f"The query generated from the javascript returned did not match the final query structure.\
The messages generated from testing the schema were:\
{errmsg}\
Please try again to find a different structure for the same query."
            print(txt)
        lux_q['_scope'] = scope
    except:
        return "The javascript generated was not valid. Please try again."
    js['query'] = lux_q
    return js

def build_query_single(client, q):

    print(q)
    js = generate(client, q)
    print(js)
    js2 = process_query(js)
    if type(js2) == str:
        return js2
    js2 = js2['query']
    # We're good
    if len(query_cache) > 128:
        query_cache.popitem()
    query_cache[q] = json.dumps(js2)
    return js2


def build_query_multi(client, q):
    print(q)
    js = generate(client, q)
    if js is None:
        # just try again?
        js = generate(client, q)
        if js is None:
            return "Could not get JSON back from the AI for that query"
    print(js)
    if 'options' in js:
        # full set of options
        for o in js['options']:
            js2 = o['q']
            js3 = process_query(js2)
            if type(js3) == str:
                # uhoh!
                # trash it?
                return js3
            # pull it up a level
            o['q'] = o['q']['query']
    # We're good
    if len(query_cache2) > 128:
        query_cache2.popitem()
    query_cache2[q] = js
    return js


##### FLASK from here on down


# Refactor to use @functools.lru_cache
query_cache = {}
query_cache2 = {}
failed_query_cache = {}

app = Flask(__name__)
@app.after_request
def add_cors(response):
    response.headers['content-type'] = "application/json"
    response.headers["access-control-allow-origin"] = "*"
    return response

@app.route('/api/translate/<string:scope>', methods=['GET'])
def make_query_single(scope):
    q = request.args.get('q', None)
    if not q:
        return ""
    elif q in query_cache:
        return query_cache[q]

    cl = client
    js = build_query_single(cl, q)
    if type(js) == str:
        js = build_query(cl, js + " " + q)
        if type(js) == str:
            failed_query_cache[q] = js
            return json.dumps({"_scope": "item", "name": f"ERROR for query: {js}"})
    jstr = json.dumps(js)
    print(f"Okay: {jstr}")
    return jstr


@app.route('/api/translate_multi/<string:scope>', methods=['GET'])
def make_query_multi(scope):
    q = request.args.get('q', None)
    if not q:
        return ""
    elif q in query_cache2:
        return query_cache2[q]

    cl = client2
    js = build_query_multi(cl, q)
    if type(js) == str:
        js = build_query2(cl, js + " " + q)
        if type(js) == str:
            failed_query_cache[q] = js
            return json.dumps({"_scope": "item", "name": f"ERROR for query: {js}"})
    jstr = json.dumps(js)
    print(f"Okay: {jstr}")
    return jstr


@app.route('/api/translate_raw/<string:scope>', methods=['GET'])
def make_query_raw(scope):
    q = request.args.get('q', None)
    if not q:
        return ""
    cl = client
    js = generate(cl, q)
    return json.dumps(js)


@app.route('/api/dump_cache', methods=['GET'])
def dump_cache():
    which = request.args.get('type', 'okay')
    if which.startswith('fail'):
        return json.dumps(failed_query_cache)
    else:
        return json.dumps(query_cache)

if __name__ == '__main__':
    app.run(debug=True, port=8443, host='0.0.0.0', ssl_context='adhoc')
