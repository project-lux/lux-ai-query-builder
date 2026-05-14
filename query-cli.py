import json
import sys
from urllib.parse import quote_plus

import requests
from google import genai
from google.genai import types

with open("google_project.txt") as fh:
    gcp_name = fh.read().strip()

LUX_HOST = "https://lux.collections.yale.edu"

with open("system-prompt-merged-short.txt") as fh:
    system_prompt = fh.read().strip()

with open("system-prompt-improve.txt") as fh:
    system_prompt_improve = fh.read().strip()


gemini_model = "gemini-3.1-flash-lite"
generated_config = types.GenerateContentConfig(
    temperature=0.8,
    top_p=0.95,
    max_output_tokens=36000,
    response_modalities=["TEXT"],
    safety_settings=[
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ],
    response_mime_type="application/json",
    thinking_config=types.ThinkingConfig(thinking_budget=2000),
    system_instruction=[types.Part.from_text(text=system_prompt)],
)

config_improve = types.GenerateContentConfig(
    temperature=0.8,
    top_p=0.95,
    max_output_tokens=36000,
    response_modalities=["TEXT"],
    safety_settings=[
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ],
    response_mime_type="application/json",
    thinking_config=types.ThinkingConfig(thinking_budget=2000),
    system_instruction=[types.Part.from_text(text=system_prompt_improve)],
)

gemini = genai.Client(
    vertexai=True,
    project=gcp_name,
    location="global",
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

uri_scopes = {
    "item": "objects",
    "work": "works",
    "agent": "people",
    "concept": "concepts",
    "place": "places",
    "event": "events",
    "set": "collections",
}

##### Functions


def generate_gemini(prompt, which="build"):
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    output = []
    cfg = generated_config if which == "build" else config_improve
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
        if False and "d" in query and query["d"]:
            print(f"SAW 'd' in {query}")

            # This is where we reach out RAG style
            # or use tool calling in the model
            # or other neat tricks
            #
            ids = ["breaks"]

            new["OR"] = [{"id": x} for x in ids]
            return new

        if query["f"] in ["height", "width", "depth", "dimension"]:
            query["v"] = float(query["v"])
        elif query["f"] in ["hasDigitalImage"]:
            query["v"] = int(query["v"])
        elif query["f"].lower() == "recordtype":
            query["v"] = query["v"].lower()
        new[query["f"]] = query["v"]
        if "c" in query:
            new["_comp"] = query["c"]
    return new


def process_js(js):
    if type(js) == list:
        js = {"options": js}
    try:
        for q in js["options"]:
            qry = q["query"]
            scope = q["scope"]
            lq = post_process(qry, scope)
            print(f"({scope})  {q['natural']}")
            print(json.dumps(lq, indent=2))
            qstr = quote_plus(json.dumps(lq, separators=(",", ":")))
            print(f"{LUX_HOST}/view/results/{uri_scopes[scope]}?q={qstr}")
            return lq
    except:
        print("Failed to process:")
        print(json.dumps(js, indent=2))
        return js


def process(user_string, change=""):
    js = generate_gemini(user_string)
    lq = process_js(js)
    if change:
        p2 = f"Query: \n{json.dumps(lq, indent=2)}\n\nImprovement: {change}"
        js2 = generate_gemini(p2, "improve")
        lq = process_js(js2)
    return lq


# process("I want books about tolkien", "no wait about C S Lewis")

process("paintings of women in paris in the 1900s", "what about just paintings of paris with people in them")
