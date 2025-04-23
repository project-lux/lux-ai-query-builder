from flask import Flask, request, redirect, make_response
import json
import requests
import functools
from urllib.parse import quote_plus

from clarity import Clarity

from jsonschema import Draft202012Validator
schemafn = "generated_schema.json"
fh = open(schemafn)
schema = json.load(fh)
fh.close()
validator = Draft202012Validator(schema)

with open('../clarity-config.json') as fh:
    config = json.load(fh)
client = Clarity(base_url=config['base_url'], instance_id=config['instance_id'], api_key=config['private_access_key'])

LUX_HOST = "https://lux.collections.yale.edu"

session = client.create_session("proxy-test")
preload = client.complete(session, "I want books about fish", config['agent_name'], parse_json=True)

def generate(prompt):
    resp = client.complete(session, prompt, config['agent_name'], parse_json=True)
    return resp['json']

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
        if js['totalItems'] > 1:
            return True
        else:
            return False
    except:
        return False


def build_query(q):

    print(q)
    js = generate(q)
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
            return txt
        lux_q['_scope'] = scope
    except:
        return "The javascript generated was not valid. Please try again."

    lux_q_str = json.dumps(lux_q)
    hits = test_hits(scope, lux_q_str)
    if not hits:
        print(lux_q_str)
        return "There were no results for that query. Please can you try again to find a different structure for the same query."

    # We're good
    if len(query_cache) > 128:
        query_cache.popitem()
    query_cache[q] = lux_q_str

    return lux_q


# Refactor to use @functools.lru_cache
query_cache = {}
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

    js = build_query(q)
    if type(js) == str:
        js = build_query(js + " " + q)
        if type(js) == str:
            failed_query_cache[q] = js
            return ""
    jstr = json.dumps(js)
    return jstr

@app.route('/api/translate_raw/<string:scope>', methods=['GET'])
def make_query_raw(scope):
    q = request.args.get('q', None)
    if not q:
        return ""
    output = generate(q)
    return json.dumps(output)

@app.route('/api/dump_cache', methods=['GET'])
def dump_cache():
    which = request.args.get('type', 'okay')
    if which.startswith('fail'):
        return json.dumps(failed_query_cache)
    else:
        return json.dumps(query_cache)

@app.route('/api/reload_system_prompt', methods=['GET'])
def reload_system_prompt():
    with open('system-prompt.txt') as fh:
        textsi_1 = fh.read().strip()

    generated_config.system_instruction=[types.Part.from_text(text=textsi_1)]
    return {"status": "ok"}

#if __name__ == '__main__':
#    app.run(debug=True)
