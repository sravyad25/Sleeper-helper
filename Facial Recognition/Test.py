import cv2
import os
import numpy as np
from mtcnn.mtcnn import MTCNN
from pycoral.utils.edgetpu import make_interpreter
import argparse
import io
from datetime import datetime
import sqlite3
from PIL import Image
import pyttsx3
import speech_recognition as sr
import openai
# from fer import FER

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def capture_image(save_path="/home/admin/Sleeper-helper-1/Facial Recognition/PiCamera_captured_images_before_cropping"):
    """Capture an image from the Raspberry Pi camera and return it as a BGR numpy array."""
    from picamera import PiCamera
    with PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.start_preview()
        stream = io.BytesIO()
        camera.capture(stream, format='jpeg')
        stream.seek(0)
        camera.stop_preview()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captured_{timestamp}.jpg"
    file_path = os.path.join(save_path, filename)
    with open(file_path, "wb") as f:
        f.write(stream.getvalue())
        
    image = Image.open(stream)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def detect_and_crop(mtcnn, image):
    """Detect faces in an image using MTCNN and return the largest detected face with added margin."""
    detection = mtcnn.detect_faces(image)[0]
    # Extract the bounding box coordinates
    bounding_box = detection['box']
    x, y, w, h = bounding_box
    
    # Add a 20% margin 
    margin = 0.2
    x_margin = int(w * margin)
    y_margin = int(h * margin)
    x = max(x - x_margin, 0)
    y = max(y - y_margin, 0)
    w = int(w * (1 + 2 * margin))
    h = int(h * (1 + 2 * margin))
    
    #image cropping 
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def preprocess_image(face, target_size=(160, 160)):
    """Preprocess the face image for embedding generation."""
    face = cv2.resize(face, target_size)
    face = face.astype('float32')
    mean = face.mean()
    std = face.std()
    if std == 0:
        face = face - mean
    else:
        face = (face - mean) / std
    return face

def run_model(interpreter, face):
    """Run the TensorFlow Lite model to generate embeddings for a given face."""
    face_input = np.expand_dims(face, axis=0)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], face_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def create_feature_vectors(mtcnn, interpreter, uploads_folder):
    """Create a dictionary of preprocessed feature vectors for all images in the uploads folder."""
    feature_vectors = {}
    image_files = os.listdir(uploads_folder)
    
    for image_file in image_files:
        image_path = os.path.join(uploads_folder, image_file)
        image = cv2.imread(image_path)
        face = detect_and_crop(mtcnn, image)
        
        if face is not None:
            preprocessed_face = preprocess_image(face)
            feature_vectors[image_file] = run_model(interpreter, preprocessed_face)
    
    return feature_vectors

def get_user_info(db_path, face_filename):
    """Retrieve the user info from the database using the face image filename."""
    # Extract just the filename without the path
    face_filename = os.path.basename(face_filename)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, sleep_or_read, ambient_noise FROM user_table WHERE face=?", (face_filename,))
    result = cursor.fetchone()
    conn.close()
    return result if result else None

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
    
    """ Exception as e:
        print(f"An error occurred with the Chat API: {str(e)}")
        return "Sorry, I couldn't understand that." """

def main(args):
    print("Loading model and MTCNN detector...")
    interpreter = make_interpreter(args.model_path)
    interpreter.allocate_tensors()
    mtcnn = MTCNN()
    
    face_embeddings_path = "/home/admin/Sleeper-helper-1/Facial Recognition/face_embeddings"
    if not os.path.exists(face_embeddings_path):
        os.makedirs("face_embeddings")
        
    uploads_folder = "/home/admin/Sleeper-helper-1/web_interface/uploads"

    image_files = os.listdir(uploads_folder)

    # Process each image
    for image_file in image_files:
        # Read the image
        image_path = os.path.join(uploads_folder, image_file)
        image = cv2.imread(image_path)

        # Detect and crop the face
        face = detect_and_crop(mtcnn, image)

        # Preprocess the cropped face
        preprocessed_face = preprocess_image(face)

        # Save the preprocessed face as a numpy array
        preprocessed_image_path = os.path.join("face_embeddings", image_file[:-4] + ".npy")
        np.save(preprocessed_image_path, preprocessed_face)

        print(f"Preprocessed and saved: {preprocessed_image_path}")
    feature_vectors = create_feature_vectors(mtcnn, interpreter, args.uploads_dir)
    print("Capturing image from PiCamera...")
    image = capture_image()
    print("Detecting and cropping face...")
    cropped_face = detect_and_crop(mtcnn, image)
    
    if cropped_face is not None:
        print("Preprocessing and embedding face...")
        face_preprocessed = preprocess_image(cropped_face)
        face_embedding = run_model(interpreter, face_preprocessed)
    
    distances = []
    filenames = []
    for name, embedding in feature_vectors.items():
        dist = np.linalg.norm(embedding - face_embedding)
        distances.append(dist)
        filenames.append(name)
        # print(f"Distance between {name} and the captured image:",
        
    import math
    min_dist = math.inf
    min_file_name = None
    for i in range(len(distances)):
        if distances[i] < min_dist:
            min_dist = distances[i]
            min_file_name = filenames[i]

    result = get_user_info(args.db_path, min_file_name)
    text = "Hello " + result[0] + "!"
    
    # detector = FER()
    # emotion, score = detector.top_emotion(Image.fromarray(cropped_face))
    
    greet_user(str(result[0]), "happy")
    
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
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .tflite model file.')
    parser.add_argument('--uploads_dir', type=str, required=True, help='Directory path of uploaded face images.')
    parser.add_argument('--embeddings_dir', type=str, required=True, help='Directory path to store face embeddings.')
    parser.add_argument('--db_path', type=str, required=True, help='Path to the SQLite database file.')
    args = parser.parse_args()
    main(args)