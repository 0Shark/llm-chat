import os
from dotenv import load_dotenv
from elevenlabs import generate, stream
from elevenlabs import set_api_key

load_dotenv()
set_api_key(os.getenv("ELEVENLABS_API_KEY"))

def speak(text: str):
  audio_stream = generate(text=text, stream=True, voice="Sally")
  stream(audio_stream)