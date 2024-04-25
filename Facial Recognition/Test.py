import cv2
import os
import numpy as np
from mtcnn.mtcnn import MTCNN
from pycoral.utils.edgetpu import make_interpreter
import argparse

# Assuming your preprocess_image and detect_and_crop functions are defined
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
        face = face - mean  # Just remove the mean if std is zero
    else:
        face = (face - mean) / std
    return face

def run_model(interpreter, face):
    """Run the TensorFlow Lite model to generate embeddings for a given face."""
    face_input = np.expand_dims(face, axis=0)  # Add batch dimension
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], face_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def main(args):
    print("Loading model and MTCNN detector...")
    interpreter = make_interpreter(args.model_path)
    interpreter.allocate_tensors()
    mtcnn = MTCNN()
    
    # Create the "face_embeddings" folder if it doesn't exist
    if not os.path.exists("/home/admin/Sleeper-helper-1/Facial Recognition/face_embeddings"):
        os.makedirs("face_embeddings")

    # Path to the "uploads" folder
    uploads_folder = "/home/admin/Sleeper-helper-1/web_interface/uploads"

    # Get the list of image files in the "uploads" folder
    image_files = os.listdir(uploads_folder)

    # Process each image
    for image_file in image_files:
        # Read the image
        image_path = os.path.join(uploads_folder, image_file)
        image = cv2.imread(image_path)

        # Detect and crop the face
        face = detect_and_crop(mtcnn, image)  # Assuming mtcnn is defined elsewhere

        # Preprocess the cropped face
        preprocessed_face = preprocess_image(face)

        # Save the preprocessed face as a numpy array
        preprocessed_image_path = os.path.join("face_embeddings", image_file[:-4] + ".npy")
        np.save(preprocessed_image_path, preprocessed_face)

        print(f"Preprocessed and saved: {preprocessed_image_path}")
    
    print("Capturing image from PiCamera...")
    # image = capture_image()
    path = "/home/admin/Sleeper-helper-1/Facial Recognition/PiCamera_captured_images_before_cropping/test_photo.jpg"
    image = cv2.imread(path)
    print("Detecting and cropping face...")
    cropped_face = detect_and_crop(mtcnn, image)
    
    if cropped_face is not None:
        print("Preprocessing and embedding face...")
        face_preprocessed = preprocess_image(cropped_face)
        face_embedding = run_model(interpreter, face_preprocessed)
    
    test = np.load("/home/admin/Sleeper-helper-1/Facial Recognition/face_embeddings/sravya2.npy")
    
    test_vector = run_model(interpreter, test)
    dist = np.linalg.norm(test_vector - face_embedding)
    print("Distance between images:", dist)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .tflite model file.')
    parser.add_argument('--uploads_dir', type=str, required=True, help='Directory path of uploaded face images.')
    parser.add_argument('--embeddings_dir', type=str, required=True, help='Directory path to store face embeddings.')
    parser.add_argument('--db_path', type=str, required=True, help='Path to the SQLite database file.')
    args = parser.parse_args()
    main(args)
