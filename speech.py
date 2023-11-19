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
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_KEY"), region=os.getenv("AZURE_SPEECH_REGION"))
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    print("Say something...")

    def handle_intermediate_result(evt):
        print("You: {}".format(evt.result.text), flush=True)

    speech_recognizer.recognizing.connect(handle_intermediate_result)

    result = speech_recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
        return False