# chat.py
# Author: Juled Zaganjori
# chat.py uses the OpenAI API to generate text based on a prompt.

import requests
import os
import base64
import threading
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from web_client import RestClient
from speech import speak_openai, transcribe_azure, transcribe_openai
from langchain.schema import SystemMessage

load_dotenv()

assistant_name = "GPT"

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


def user_input_thread():
    global user_input
    user_input = input("You: ")


def start_chat(speech: bool = False):
  """
  Starts a chat with the LLM.
  """
  global assistant_name, assistant_state
  llm = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=os.getenv("OPENAI_KEY"))
  tools = [search_web]
  memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
  system_message_speech = "You are a computer terminal based assistant that can answer questions and perform tasks. Your name is " + assistant_name + ". Please try to act as humane as possible this means you should include pauses and other human-like behaviors (such as \"um\" and \"uh\"), stuttering, and other natural language patterns. Keep your responses short and to the point but always fully answer the user's question. Don't advertise third-party products or services that are not directly related to the user's query. No profanity or explicit content. Your output will go through a text-to-speech engine to be spoken to the user so make sure you instruct it properly."
  system_message_text = "You are a computer terminal based assistant that can answer questions and perform tasks. Your name is " + assistant_name + ". Please try to act as humane as possible"
  
  system_message = system_message_speech if speech else system_message_text
  
  agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION , verbose=False, memory=memory, handle_parsing_errors=True, agent_kwargs={"system_message": system_message})

  if speech:
    print("Speak something to start the conversation...")
    
  while True:
    if speech:
      user_input = transcribe_openai()
    else:
      user_input = input("You: ")
      
    if not user_input:
      print("Please provide an input.")
      continue
    
    response = agent_chain(user_input)
    print(assistant_name + ": " + response["output"])
    if speech:
      speak_openai(response["output"])