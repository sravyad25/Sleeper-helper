import argparse
import numpy as np
import cv2
import io
import os
from datetime import datetime
# import pyttsx3
# import speech_recognition as sr
import sqlite3
from PIL import Image
from mtcnn.mtcnn import MTCNN
from pycoral.utils.edgetpu import make_interpreter
# from fer import FER
# import nltk
# from nltk.tokenize import word_tokenize

# Initialize text-to-speech engine and speech recognizer
# engine = pyttsx3.init()
# recognizer = sr.Recognizer()

# Download necessary NLTK data for tokenization
# nltk.download('punkt')

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
    detections = mtcnn.detect_faces(image)
    if detections:
        x, y, width, height = detections[0]['box']
        margin = int(max(width, height) * 0.2)
        x_new = max(0, x - margin)
        y_new = max(0, y - margin)
        width_new = width + 2 * margin
        height_new = height + 2 * margin
        return image[y_new:y_new+height_new, x_new:x_new+width_new]
    return None

def preprocess_image(face, target_size=(160, 160)):
    """Preprocess the face image for embedding generation."""
    face = cv2.resize(face, target_size)
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    return (face - mean) / std

def run_model(interpreter, face):
    """Run the TensorFlow Lite model to generate embeddings for a given face."""
    face_input = np.expand_dims(face, axis=0)  # Add batch dimension
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], face_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def load_embeddings(embeddings_path):
    """Load precomputed embeddings from the specified directory."""
    embeddings = {}
    for emb_file in os.listdir(embeddings_path):
        if emb_file.endswith('.npy'):
            user_id = os.path.splitext(emb_file)[0]
            embeddings[user_id] = np.load(os.path.join(embeddings_path, emb_file))
    return embeddings

def save_embedding(embedding, face_filename, embeddings_path):
    """Save a computed embedding to disk."""
    np.save(os.path.join(embeddings_path, f'{face_filename}.npy'), embedding)

def precompute_embeddings_for_uploads(upload_dir, embeddings_dir, interpreter, mtcnn):
    """Compute and save embeddings for all face images in the upload directory."""
    for filename in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, filename)
        image = cv2.imread(file_path)
        cropped_face = detect_and_crop(mtcnn, image)
        if cropped_face is not None:
            face_preprocessed = preprocess_image(cropped_face)
            embedding = run_model(interpreter, face_preprocessed)
            face_filename = filename.split('.')[0]
            save_embedding(embedding, face_filename, embeddings_dir)

def get_user_info(db_path, face_filename):
    """Retrieve the user info from the database using the face image filename."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, sleep_or_read, ambient_noise FROM user_table WHERE face=?", (face_filename,))
    result = cursor.fetchone()
    conn.close()
    return result if result else None

def main(args):
    """Main function to initialize components and run the face recognition loop."""
    interpreter = make_interpreter(args.model_path)
    interpreter.allocate_tensors()
    mtcnn = MTCNN()

    if not os.listdir(args.embeddings_dir):
        print("Precomputing uploads embeddings...")
        precompute_embeddings_for_uploads(args.uploads_dir, args.embeddings_dir, interpreter, mtcnn)
    
    print("Loading Embeddings...")
    embeddings = load_embeddings(args.embeddings_dir)
    
    print("Capturing Image from PiCamera...")
    image = capture_image()
    
    print("Cropping Face...")
    cropped_face = detect_and_crop(mtcnn, image)
    if cropped_face is not None:
        print("Preprocessing Face...")
        face_preprocessed = preprocess_image(cropped_face)
        print("Calculating Feature Vector...")
        face_embedding = run_model(interpreter, face_preprocessed)
        min_dist = float('inf')
        face_filename_closest = None
        
        print("Calculating distances...")
        file_names = []
        dists = []
        for face_filename, embedding in embeddings.items():
            dist = np.linalg.norm(embedding - face_embedding)
            file_names.append(face_filename)
            dists.append(dist)
            if dist < min_dist:
                min_dist = dist
                face_filename_closest = face_filename
                
        print("Person identified...")
        print(min_dist)
        print(face_filename_closest)
        print(file_names)
        print(dists)
        
        if face_filename_closest:
            print("Getting User Info...")
            user_info = get_user_info(args.db_path, face_filename_closest + '.jpg')
            if user_info:
                user_name, sleep_or_read, ambient_noise = user_info
                print(f"Detected {user_name} with distance {min_dist}")
                    # detector = FER()
                    # emotion, score = detector.top_emotion(Image.fromarray(cropped_face))
                    # print(f"Emotion: {emotion}")
                    # engine.say(f"Hello {user_name}, you seem {emotion}. Would you like to {sleep_or_read}? I can adjust the ambient noise to {ambient_noise} for you.")
                    # engine.runAndWait()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .tflite model file.')
    parser.add_argument('--uploads_dir', type=str, required=True, help='Directory path of uploaded face images.')
    parser.add_argument('--embeddings_dir', type=str, required=True, help='Directory path to store face embeddings.')
    parser.add_argument('--db_path', type=str, required=True, help='Path to the SQLite database file.')
    args = parser.parse_args()
    main(args)

