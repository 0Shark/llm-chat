import os
import speech_recognition as sr
from openai import OpenAI
from dotenv import load_dotenv
from elevenlabs import generate, stream
from elevenlabs import set_api_key
import azure.cognitiveservices.speech as speechsdk
import pyaudio

load_dotenv()
set_api_key(os.getenv("ELEVENLABS_API_KEY"))

openaiClient = OpenAI(
    api_key=os.getenv("OPENAI_KEY")
)

r = sr.Recognizer()

def speak_eleven(text: str):
  audio_stream = generate(text=text, stream=True, voice="Sally")
  stream(audio_stream)
  
  
def speak_openai(text: str):
    audio_response = openaiClient.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
        response_format="opus"
    )
    audio_stream = audio_response.response.iter_bytes(1024)
    stream(audio_stream)


def transcribe_azure():
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_KEY"), region=os.getenv("AZURE_SPEECH_REGION"))
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    print("Say something...")

    def handle_intermediate_result(evt):
        print("You: {}".format(evt.result.text), end="\r", flush=True)

    speech_recognizer.recognizing.connect(handle_intermediate_result)

    result = speech_recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
        return False
    
    
def transcribe_openai():
    # Record audio
    print("Speak something...")
    with sr.Microphone() as source:
        audio = r.listen(source)
    
    # Detect speech
    try:
        text = r.recognize_google(audio)
    except sr.UnknownValueError:
        print("Could not understand audio")
        return False
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        return False
    
    # save audio to file
    with open("audio.wav", "wb") as f:
        f.write(audio.get_wav_data())
        
    audio = open("audio.wav", "rb")    
    
    print("Transcribing...")
    transcription = openaiClient.audio.transcriptions.create(
    model="whisper-1", 
    file=audio,
    response_format="text"
    )
    return transcription