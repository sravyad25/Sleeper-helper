import pyttsx3
import speech_recognition as sr
import openai

engine = pyttsx3.init()

def speak(text):
    """Uses pyttsx3 to convert text to speech."""
    engine.say(text)
    engine.runAndWait()
    
def listen():
    """Listens to microphone input and converts it to text using speech recognition."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        spoken_text = recognizer.recognize_google(audio)
        print("You said: " + spoken_text)
        return spoken_text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def greet_user(name, emotion):
    """Generates an initial greeting message based on the user's name and perceived emotion."""
    greeting = f"Hello {name}, you seem {emotion}. Would you like to talk about your day?"
    speak(greeting)

def chat_with_chatgpt(prompt):
    """Sends text to ChatGPT and gets a response."""
    openai.api_key = 'sk-proj-zoAfO3UXTjomKCdt5fjOT3BlbkFJF0a9EdfPWwoLkBHspMgg'  # Make sure to replace this with your actual API key
    completion = openai.Completion.create(model="davinci-002", prompt=prompt, max_tokens=10)
    return completion.choices[0].text


def main():
    greet_user(str("Junhao"), "happy")
    
    session_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly."
    while True:
        user_input = listen()
        if user_input and user_input != "thank you":
            session_prompt += f"\nHuman: {user_input}\nAI:"
            response_text = chat_with_chatgpt(session_prompt)
            session_prompt += f" {response_text}"
            speak(response_text)
        else:
            break
    
    
    speak("Would you like to sleep")

main()