import time
import json
import logging
from clarity import Clarity

logger = logging.getLogger("lux-clarity")

with open('../clarity-config.json') as fh:
    config = json.load(fh)

client = Clarity(base_url=config['base_url'], instance_id=config['instance_id'], api_key=config['private_access_key'])

prompt = "I want paintings of professors who discovered fossils"

start = time.time()
session_id = client.create_session("test")
print(time.time()-start)
start = time.time()
completion = client.complete(session_id, prompt, config['agent_name'], parse_json=True)
print(time.time()-start)
print(completion)
