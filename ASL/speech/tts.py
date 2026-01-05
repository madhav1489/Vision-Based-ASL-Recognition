import pyttsx3

engine = pyttsx3.init()

def speak_text(text):
    if not text.strip():
        return
    engine.say(text)
    engine.runAndWait()
