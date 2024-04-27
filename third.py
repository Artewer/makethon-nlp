from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine import SimpleChatEngine
from dotenv import load_dotenv
import os
load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-0613')

def getFollowupQuestions(query, answer):
    prompt = f"Given the query '{query}', the answer is '{answer}'. What follow-up questions might a user have? Provide only one question and be different from the previous one."
    chat_engine = SimpleChatEngine.from_defaults(llm=llm)
    questions = []
    for i in range(4):
        response = chat_engine.chat(prompt)
        questions.append(response.response)
    return questions

if __name__ == '__main__':
    print("third")