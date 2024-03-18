from langchain.llms import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage

import os
import openai
from dotenv import load_dotenv

load_dotenv()

llm = AzureChatOpenAI(
    azure_deployment = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")
)

prompt = "Explain the transcendentalist school of thought at a third grade reading level"

msg = HumanMessage(content=prompt)

# Call the API
r = llm.invoke([msg])

# Print the response
print(r.content)
