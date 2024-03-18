import openai
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version = os.getenv("OPENAI_API_VERSION")
)

prompt = "Explain the transcendentalist school of thought at a third grade reading level"

response = client.chat.completions.create(
    model = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME"),
    messages = [{"role" : "assistant", "content" : prompt}],
)

print(response)
print(response.choices[0].message.content)
