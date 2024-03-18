import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("OPENAI_API_VERSION")
RESOURCE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_ID = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")

url = RESOURCE_ENDPOINT + "/openai/deployments/" + DEPLOYMENT_ID + "/chat/completions?api-version=" + API_VERSION

prompt = "Explain the transcendentalist school of thought at a third grade reading level"
r = requests.post(url, headers={"api-key": API_KEY}, json={"messages":[{"role": "assistant", "content": prompt}]})

print(json.dumps(r.json(), indent=2))
print(url)
