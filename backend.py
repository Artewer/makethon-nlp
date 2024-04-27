from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import (
    load_index_from_storage,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import os
# Load environment variables from .env file
from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import second

load_dotenv()
# Retrieve the OpenAI API key from the environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
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

host="0.0.0.0"
socket = 8000

app = FastAPI()

html = open("web/index.html", "r").read()
@app.get("/")
async def get():
    return HTMLResponse(html)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"ShalopAI: {agent.chat(data).response}")

import uvicorn

if __name__ == '__main__':
    uvicorn.run(app, host=host, port=socket)
