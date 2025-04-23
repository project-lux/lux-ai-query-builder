import json
import requests
import logging

logger = logging.getLogger("lux-clarity")
with open('../clarity-config.json') as fh:
    config = json.load(fh)

headers = {
    "Content-Type": "application/json",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Accept": "text/plain",
    "X-AGENT-ACCESS-TOKEN": config['private_access_key']
} 
api = f"{config['base_url']}/instances/{config['instance_id']}/"

def create_session(session_name):
    url = f"{api}/sessions"
    body = {"name": session_name}
    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        response_body = response.json()
        session_id = response_body.get("sessionId")
        return session_id
    else:
        raise Exception(f"Failed to create session with status code {response.status_code}")

def complete(session_id, prompt):
    url = f"{api}/completions"
    body = {
        "user_prompt": prompt,
        "agent_name": config['agent_name'],
        "session_id": session_id
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        response_body = response.json()
        try:
            agent_response = response_body.get("content", [{}])[0].get("value")
        except IndexError as e:
            print(f"IndexError occurred: {e}")
            agent_response = response_body.get("text")
        return agent_response
    else:
        raise Exception(f"Failed to request a completion with status code {response.status_code}")


prompt = "I want paintings of professors who discovered fossils"

session_id = create_session("test")
completion_response = complete(session_id, prompt)

