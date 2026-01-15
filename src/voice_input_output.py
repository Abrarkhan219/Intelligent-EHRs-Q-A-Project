import speech_recognition as sr
import pyttsx3

# Function to convert speech to text
def voice_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak now...")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)  # Using Google's speech recognition API
            print(f"You said: {query}")
            return query
        except Exception as e:
            print("Sorry, I couldn't understand. Please try again.")
            return None

# Function to convert text to speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()