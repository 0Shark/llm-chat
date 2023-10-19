import os
from dotenv import load_dotenv
from elevenlabs import generate, stream
from elevenlabs import set_api_key
import azure.cognitiveservices.speech as speechsdk

load_dotenv()
set_api_key(os.getenv("ELEVENLABS_API_KEY"))

def speak(text: str):
  audio_stream = generate(text=text, stream=True, voice="Sally")
  stream(audio_stream)

def transcribe():
    print("Initializing speech recognition...")
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_KEY"), region=os.getenv("AZURE_SPEECH_REGION"))
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    print("Say something...")
    result = speech_recognizer.recognize_once_async().get()
    print("You said: " + result.text)
    return result.text