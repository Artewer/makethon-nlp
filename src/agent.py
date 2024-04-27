from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import (
    load_index_from_storage,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.tools import QueryEngineTool
import os

def read_folder(path):
    # Walk through all files in the given path and return a list of documents
    documents = []
    all_files = []
    for root, dirs, files in os.walk(path):
        all_files.append(files)
        for file in files:
            document_path = os.path.join(root, file)
            documents.append(SimpleDirectoryReader(input_files=[document_path]).load_data())

    indexes = []
    for i in documents:
        indexes.append(VectorStoreIndex.from_documents(documents=i))

    return indexes, files



def create_agent():  
    indexes, files = read_folder('data/')

    llm = OpenAI(model="gpt-3.5-turbo-0613")

    tools = [QueryEngineTool.from_defaults(
        indexes[i].as_query_engine(), name=f"engine_{i}", description=f"Provides information about {files[i]}"
    ) for i in range(len(indexes))]

    agent = OpenAIAgent.from_tools(
        tools=tools,
        llm=llm,
        verbose=True
    )
    print(agent)
    return agent
