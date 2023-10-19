# chat.py
# Author: Juled Zaganjori
# chat.py uses the OpenAI API to generate text based on a prompt.

import requests
import os
import base64
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from web_client import RestClient
from speech import speak

load_dotenv()

# Web Search with SERP API
@tool
def search_web(query: str) -> str:
  """
  Searches the internet for additional information on the given query in case the LLM can not provide an answer.
  Specifically useful when you need to answer questions about current events or other topics that are not yet in the LLM's training data.
  """
  client = RestClient(os.getenv("SERP_API_LOGIN"), os.getenv("SERP_API_PASSWORD"))
  post_data = dict()
  post_data[len(post_data)] = dict(
      language_code="en",
      location_code=2840,
      keyword=query
  )
  response = client.post("/v3/serp/google/organic/live/regular", post_data)
  web_result = ""
  if response["status_code"] == 20000:
      for task in response["tasks"]:
          if task["status_code"] == 20000:
              for i, result in enumerate(task["result"][0]["items"]):
                  if result["type"] == "organic" and i < 5:
                      web_result += "[Title:" + result["title"] + ";URL:" + result["url"] + ";Description:" + result["description"] + "]"
          else:
              web_result = "[error. Code: %d Message: %s" % (task["status_code"], task["status_message"]) + "]"
  else:
      web_result = "[error. Code: %d Message: %s" % (response["status_code"], response["status_message"]) + "]"

  return web_result

llm = ChatOpenAI(model="gpt-4", temperature=0.2)
tools = [search_web]
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=False, memory=memory, handle_parsing_errors=True)

def start_chat():
  """
  Starts a chat with the LLM.
  """
  print("Input something to start the conversation. Type 'exit' to quit.")
  while True:
    user_input = input("User: ")
    if user_input == "exit":
      break
    response = agent_chain(user_input)
    print("AI: " + response["output"])
    speak(response["output"])