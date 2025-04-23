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

def walk_query(query):
    # This is the real LUX structure
    for term in ['aboutConcept', 'classification']:
        if term in query:
            # Trap this and send to resolver
            if 'name' in query[term] and ' ' in query[term]['name']:
                desc = query[term]['name']
                clause = resolve(desc)
                if clause:
                    del query[term]['name']
                    query[term][clause['field']] = clause['value']
                return
    for b in ['AND', 'OR', 'NOT']:
        if b in query:
            for sub in query[b]:
                walk_query(sub)
            return
    for k, sub in query.items():
        if not k.startswith('_') and type(sub) == dict:
            walk_query(sub)

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

def test_response(q, js, attempt=1):

    query = js['query']
    scope = js['scope']
    q2 = post_process(query)

    try:
        js3 = {"q": q2, "scope": scope}
        errs = list(validator.iter_errors(js3))
        if errs:
            for e in errs:
                print(f"  /{'/'.join([str(x) for x in e.absolute_path])} --> {e.message} ")
            raise ValueError(f"schema validation failed for {q2}")
        q2['_scope'] = scope
        jstr = json.dumps(q2)
    except:
        raise ValueError(f"invalid json in {q2}")

    # Test hits
    okay = test_hits(scope, jstr)
    if okay:
        if len(query_cache) > 128:
            query_cache.popitem()
        query_cache[q] = jstr
    elif attempt == 1:
        print(f"Failed hits for {jstr}, walking")
        # No direct hits, look for `aboutConcept` and run the resolver
        walk_query(q2)
        jstr2 = json.dumps(q2)
        if jstr2 != jstr:
            # At least one change
            okay = test_hits(scope, jstr2)
            if okay:
                query_cache[q] = jstr2
            return jstr2
        else:
            raise ValueError(f"No hits for {q2}")
    else:
        failed_query_cache[q] = jstr
    return jstr


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

    js = generate(q)
    try:
        jstr = test_response(q, js)
    except ValueError as e:
        print(e)

        output = generate(q)
        print(f"Attempt 2...")
        try:
            jstr = test_response(q, output, attempt=2)
        except ValueError as e:
            print(e)
            return error_q.replace("ERROR", str(e))

    print(jstr)
    return jstr

@app.route('/api/translate_raw/<string:scope>', methods=['GET'])
def make_query_raw(scope):
    q = request.args.get('q', None)
    if not q:
        return ""
    output = generate(q)
    return output

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
