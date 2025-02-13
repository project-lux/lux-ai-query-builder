from flask import Flask, request, redirect, make_response
import json
import requests
import functools
from urllib.parse import quote_plus

import vertexai.generative_models as generative_models
from vertexai.generative_models import GenerativeModel, Part
from google import genai
from google.genai import types

from jsonschema import Draft202012Validator
schemafn = "generated_schema.json"
fh = open(schemafn)
schema = json.load(fh)
fh.close()
validator = Draft202012Validator(schema)

with open('google_project.txt') as fh:
    gcp_name = fh.read().strip()

client = genai.Client(
      vertexai=True,
      project=gcp_name,
      location="us-central1",
)

with open('system-prompt.txt') as fh:
    textsi_1 = fh.read().strip()

model = "gemini-2.0-flash-001"
# model = "gemini-2.0-pro-exp-02-05"
# model = "gemini-1.5-pro-002"

generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 0.95,
    max_output_tokens = 8192,
    response_modalities = ["TEXT"],
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    response_mime_type = "application/json",
    system_instruction=[types.Part.from_text(text=textsi_1)],
)

def generate(prompt):
  contents = [
    types.Content(
      role="user",
      parts=[
        types.Part.from_text(text=prompt)
      ]
    )
  ]
  output = []
  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    output.append(chunk.text)
  return ''.join(output)

error_q = json.dumps({"_scope": "item", "text": "ERROR"})

def post_process(query):
    new = {}
    if 'p' in query:
        # BOOL
        new[query['f']] = [post_process(x) for x in query['p']]
    elif 'r' in query:
        new[query['f']] = post_process(query['r'])
    else:
        new[query['f']] = query['v']
        if 'c' in query:
            new['_comp'] = query['c']
    return new

def test_response(q, output, attempt=1):
    output = output.strip()
    if not output.endswith('}'):
        output = output + "}"
    try:
        js = json.loads(output)
        query = js['query']
        scope = js['scope']
    except:
        raise ValueError("invalid JSON in response from AI")

    q2 = post_process(query)

    try:
        js3 = {"q": q2, "scope": scope}
        errs = list(validator.iter_errors(js3))
        if errs:
            for e in errs:
                print(f"  /{'/'.join([str(x) for x in e.absolute_path])} --> {e.message} ")
            raise ValueError("schema validation failed")
        q2['_scope'] = scope
        jstr = json.dumps(q2)
    except:
        raise ValueError("invalid json in Q")

    # Test hits
    okay = test_hits(scope, jstr)
    if okay:
        if len(query_cache) > 128:
            query_cache.popitem()
        query_cache[q] = jstr
    elif attempt == 1:
        raise ValueError("Try again...")
    return jstr


def test_hits(scope, query):
    encq = quote_plus(query)
    url = f"http://localhost:8080/api/search-estimate/{scope}?q={encq}"
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

app = Flask(__name__)
@app.after_request
def add_cors(response):
    response.headers['content-type'] = "application/json"
    response.headers["access-control-allow-origin"] = "*"
    return response

@app.route('/api/translate/<string:scope>', methods=['GET'])
def make_Q(scope):
    q = request.args.get('q', None)
    if not q:
        return ""
    elif q in query_cache:
        return query_cache[q]

    output = generate(q)
    try:
        jstr = test_response(q, output)
    except:
        output = generate(q)
        print(f"Attempt 2: {output}")
        try:
            jstr = test_response(q, output, attempt=2)
        except ValueError as e:
            return error_q.replace("ERROR", str(e))

    print(jstr)
    return jstr

@app.route('/api/dumpcache', methods=['GET'])
def dump_cache():
    return json.dumps(query_cache)


if __name__ == '__main__':
    app.run(debug=True)
