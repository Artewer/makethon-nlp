from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine import SimpleChatEngine
import json

llm = OpenAI(model='gpt-4-turbo')


def getFollowupQuestions(query, answer):
    prompt = f"Given the query '{query}' and the answer is '{answer}'. What follow-up questions might a user have? Only write a question:"
    chat_engine = SimpleChatEngine.from_defaults(llm=llm)
    questions_json = {"questions": []}
    for i in range(4):
        response = chat_engine.chat(prompt)
        questions_json["questions"].append(response.response)
    return json.dumps(questions_json)
