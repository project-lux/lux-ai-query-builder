from flask import Flask, request, redirect, make_response
import json
import requests
import functools
from urllib.parse import quote_plus
import copy

from clarity import Clarity

from jsonschema import Draft202012Validator
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


###
### NOTE WELL
###
### Before production, UI needs to send a session token
### Then we need to generate a new session per react session
### Otherwise, "I want my previous query" will return the previous
### user's query in the same session

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


def generate(client, prompt):
    resp = client.complete(prompt, parse_json=True)
    try:
        return resp['json']
    except:
        return None

def post_process(query):
    new = {}
    if 'p' in query:
        # BOOL
        new[query['f']] = [post_process(x) for x in query['p']]
    elif 'r' in query:
        new[query['f']] = post_process(query['r'])
    else:
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
        lux_q = post_process(js['query'])
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

def build_query(client, q):

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


def build_query2(client, q):

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


def fetch_records(scope, query):
    encq = quote_plus(query)
    url = f"{LUX_HOST}/api/search/{scope}?q={encq}"
    try:
        resp = requests.get(url)
        js = resp.json()
        recs = []
        if js['totalItems'] >= 1:
            for u in js['orderedItems']:
                r2 = requests.get(u['id'])
                rjs = r2.json()
                if rjs is not None:
                    try:
                        del rjs['_links']
                    except:
                        pass
                    recs.append(rjs)
        else:
            print(f"No hits in {query}\n{js}")
        return recs
    except Exception as e:
        print(e)
        return None


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
def make_query(scope):
    q = request.args.get('q', None)
    if not q:
        return ""
    elif q in query_cache:
        return query_cache[q]

    cl = client

    js = build_query(cl, q)
    if type(js) == str:
        js = build_query(cl, js + " " + q)
        if type(js) == str:
            failed_query_cache[q] = js
            return json.dumps({"_scope": "item", "name": f"ERROR for query: {js}"})
    jstr = json.dumps(js)
    print(f"Okay: {jstr}")
    return jstr


@app.route('/api/translate_multi/<string:scope>', methods=['GET'])
def make_query2(scope):
    q = request.args.get('q', None)
    if not q:
        return ""
    elif q in query_cache2:
        return query_cache2[q]

    cl = client2
    js = build_query2(cl, q)
    if type(js) == str:
        js = build_query2(cl, js + " " + q)
        if type(js) == str:
            failed_query_cache[q] = js
            return json.dumps({"_scope": "item", "name": f"ERROR for query: {js}"})
    jstr = json.dumps(js)
    print(f"Okay: {jstr}")
    return jstr


@app.route('/api/rag/', methods=['GET'])
def rag_query():
    q = request.args.get('q', None)
    if not q:
        return ""
    js = build_query2(client2, q)
    if type(js) == str:
        js = build_query2(client2, js + " " + q)
        if type(js) == str:
            failed_query_cache[q] = js
            return "Could not create a database query for that"
    q = js['options'][0]['q']
    # Execute the query and fetch the first 10 hits
    rq = copy.deepcopy(q)
    del rq['_scope']
    rqstr = json.dumps(rq)
    recs = fetch_records(q['_scope'], rqstr)
    if recs:
        # send to AI as context for original question.
        query = []
        query.append(f"The user asked: {q}")
        query.append(f"You generated this query to try and answer it: {js['options'][0]['rewritten']}")
        query.append(f"The answers from the database are the following records in Linked Art JSON:")
        for r in recs:
            query.append(json.dumps(r))
        query.append("Using these records, please answer the user's original question")
        qstr = "\n".join(query)

    r = client3.complete(prompt)
    return r



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

@app.route('/api/dump_sessions', methods=['GET'])
def dump_sessions():
    which = request.args.get('model', '')
    if which == 'claude':
        cl = client2
    elif which == 'dev':
        cl = client3
    elif which == 'claude-dev':
        cl = client4
    else:
        cl = client
    return json.dumps([x.history for x in cl.sessions.values()])

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
