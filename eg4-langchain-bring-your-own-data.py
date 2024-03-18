from langchain.llms import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA

import os
import openai
from dotenv import load_dotenv

load_dotenv()

llm = AzureChatOpenAI(
    azure_deployment = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")
)

prompt = "Tell me about the latest Ant Man movie. When was it released? What is it about?"

# unaugmented
msg = HumanMessage(content=prompt)
r = llm.invoke([msg])
unaugmented_response = r.content

# augmented
data_dir = "labs/03-orchestration/02-Embeddings/data/movies"
documents = DirectoryLoader(path=data_dir, glob="*.md", show_progress=True, loader_cls=UnstructuredMarkdownLoader).load()
chain = load_qa_chain(llm)
result = chain.invoke({'input_documents': documents, 'question': prompt})
augmented_response = result['output_text']

# augmented with embeddings
# this is more scalable. It pre-calculates vectors from the input documents, searches them, and provides only relavant portions to the llm
# this reduces the number of tokens submitted

embeddings_model = AzureOpenAIEmbeddings(    
    azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version = os.getenv("OPENAI_EMBEDDING_API_VERSION"),
    model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
)

documents = DirectoryLoader(path=data_dir, glob="*.md", show_progress=True, loader_cls=UnstructuredMarkdownLoader).load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
document_chunks = text_splitter.split_documents(documents) # plain old list!

qdrant = Qdrant.from_documents(
    document_chunks,
    embeddings_model,
    location=":memory:",
    collection_name="movies",
)

retriever = qdrant.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

prompt_vague = "Tell me about the latest MCU movie. When was it released? What is it about?"
result_embedded_vague = qa.invoke(prompt_vague)

# comparisons
msg = HumanMessage(content=prompt_vague)
r = llm.invoke([msg])
result_vague = r.content


print(result_embedded_vague['result'])
print(result_vague)
