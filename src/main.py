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
from websocket import ConnectionChatManager

from followup_questions import getFollowupQuestions
import json


from personalization import get_info_customer

load_dotenv()
# Retrieve the OpenAI API key from the environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

app = FastAPI()
agent = a.create_agent()

manager = ConnectionChatManager()


html = open("web/index.html", "r").read()
@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    while True:
        try:
            data = await websocket.receive_text()
            persona, class_persona = get_info_customer(data)
            prompt = (
            "You are a personalized chatbot to answer questions about electrical vehicles. Your target is to make the user buy an electrical vehicle at whatever cost."
            "Adjust your behaviour to a specified persona. Don't reveal the knowledge about a person."
            f"Here is the personal data:\n{persona}\n\n"
            f"Here is a possible characteristic of a person:\n{class_persona}\n\n"
            f"Here is the user query: \n{data}\n"
            )
            await manager.send_text(agent.chat(prompt).response, websocket)
            ask_prompt = "Ask one question to find out what are the user's preferences in electric vehicles"
            await manager.send_text(agent.chat(ask_prompt).response, websocket)
            await manager.send_text(getFollowupQuestions(query=data, answer=prompt), websocket)
        except WebSocketDisconnect:
                manager.disconnect(websocket)
                # Reset all values in person.json to empty strings
                with open('person.json', 'r+') as json_file:
                    data = json.load(json_file)
                    for key in data.keys():
                        data[key] = ""
                    json_file.seek(0)  # Reset file position to the beginning.
                    json.dump(data, json_file, indent=4)
                    json_file.truncate()  # Remove any leftover data.

@app.websocket("/ws_test")
async def websocket_endpoint_new(websocket: WebSocket):
    await manager.connect(websocket)
    while True:
        try:
            data = await websocket.receive_text()
            await manager.send_text('hello world', websocket)
        except WebSocketDisconnect:
                manager.disconnect(websocket)

            


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
