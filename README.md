# lux-ai-query-builder
A LLM powered advanced search query builder


## Running it

Create a file called "google_project.txt"
Put the name of your google project in it.
e.g. `echo "name of your project" > google_project.txt`

Then:
```
export FLASK_APP=ai-proxy.py
export FLASK_DEBUG=True
flask run --host localhost
```

## Using it

Make a request with a `q` parameter to `/api/translate/{scope}`. The scope actually doesn't matter at the moment.

`curl http://127.0.0.1:5000/api/translate/item\?q=17th+century+italian+paintings`

will return

```json
{"AND": [{"producedDate": "1600-01-01T00:00:00", "_comp": ">="}, {"producedDate": "1700-01-01T00:00:00", "_comp": "<"}, {"producedAt": {"partOf": {"name": "Italy"}}}, {"classification": {"name": "painting"}}], "_scope": "item"}
```

if your AI is feeling like it at the time.
