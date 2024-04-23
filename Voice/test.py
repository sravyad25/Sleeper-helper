import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize
import pyttsx3

nltk.download('punkt')

# Define the classify_sentiment function
def classify_sentiment(word):
    positive_words = ['happy','bright', 'nice','wonderful','lucky','perfect','splendid','outstanding','fabulous','extraordinary','delight','delightful','marvelous']
    negative_words = ['rough', 'tough', 'bad','sad','unhappy','upset','depressing', 'awful', 'hard', 'terrible', 'difficult'] # Add more negative words as needed
    
    if word.lower() in positive_words:
        return "happy"
    elif word.lower() in negative_words:
        return "unhappy"
    else:
        return None

# Define the classify_day function
def classify_day(text):
    tokens = word_tokenize(text)
    sentiment_labels = [classify_sentiment(word) for word in tokens]
    happy_count = sentiment_labels.count("happy")
    unhappy_count = sentiment_labels.count("unhappy")
    
    if happy_count > unhappy_count:
        return "happy day"
    elif unhappy_count > happy_count:
        return "unhappy day"
    else:
        return "neutral day"

engine = pyttsx3.init()
recognizer = sr.Recognizer()

# Capture audio from the microphone
with sr.Microphone() as source:
    print("Listening... Speak something.")
    recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
    audio = recognizer.listen(source)

# Recognize speech using Google Speech Recognition
try:
    print("Recognizing...")
    text = recognizer.recognize_google(audio)
    print("You said:", text)

    # Classify the day based on the spoken words
    day_classification = classify_day(text)
    print("Today is a", day_classification)

    engine.say("Today is a " + day_classification)
    engine.runAndWait()

except sr.UnknownValueError:
    print("Sorry, I couldn't understand what you said.")
    engine.say("Sorry, I couldn't understand what you said.")
    engine.runAndWait()
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
    engine.say("Could not request results from Google Speech Recognition service.")
    engine.runAndWait()