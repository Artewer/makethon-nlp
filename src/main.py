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
import uvicorn
import agent as a

load_dotenv()
# Retrieve the OpenAI API key from the environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

app = FastAPI()

html = open("web/index.html", "r").read()
@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    agent = a.create_agent()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"ShalopAI: {agent.chat(data).response}")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
