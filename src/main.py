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

from followup_questions import getFollowupQuestions

from personalization import get_info_customer

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
        persona, class_persona = get_info_customer(data)
        prompt = (
        "You are a personalized chatbot to answer questions about electrical vehicles. Your target is to make the user buy an electrical vehicle at whatever cost."
        "Adjust your behaviour to a specified persona. Don't reveal the knowledge about a person."
        f"Here is the personal data:\n{persona}\n\n"
        f"Here is a possible characteristic of a person:\n{class_persona}\n\n"
        f"Here is the user query: \n{data}\n"
        )
        await websocket.send_text(f"ShalopAI: {agent.chat(prompt).response}")
        ask_prompt = "Ask one question to find out what are the user's preferences in electric vehicles"
        await websocket.send_text(agent.chat(ask_prompt).response)
        await websocket.send_text(getFollowupQuestions(query=data, answer=prompt))
    

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
