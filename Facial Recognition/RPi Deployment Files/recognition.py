import argparse
import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter

def run_model(interpreter, face):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    face_input = np.expand_dims(face, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], face_input)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data
#preprocess this image
from mtcnn import MTCNN
def detect_and_crop(mtcnn, image):
    detection = mtcnn.detect_faces(image)[0]
    #TODO
    
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
    
    # bounding box in the original image
    #show_bounding_box(image, (x, y, w, h))
    
    return cropped_image, (x, y, w, h)

import cv2
import picamera
import numpy as np
import io
def capture_image():
    # Instrctor note: this can be directly taken from the PiCamera documentation
    # Create the in-memory stream
    stream = io.BytesIO()
    with picamera.PiCamera() as camera:
        camera.capture(stream, format='jpeg')
        
    # Construct a numpy array from the stream
    data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
    
    # "Decode" the image from the array, preserving colour
    image = cv2.imdecode(data, 1)
    
    # OpenCV returns an array with data in BGR order. 
    # The following code invert the order of the last dimension.
    image = image[:, :, ::-1]
    return image

def pre_process(face, required_size=(160, 160)):
    ret = cv2.resize(face, required_size)
    ret = ret.astype('float32')
    mean, std = ret.mean(), ret.std()
    ret = (ret - mean) / std
    return ret

def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m', '--model', required=True, help='File path of .tflite file.')
    parser.add_argument(
        '-pf', '--preprocessed_faces', required=True, help='File path of preprocessed faces directory.')
    args = parser.parse_args()

    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    
    mtcnn = MTCNN()
    new_image = capture_image() #take image
    cropped_image, _ = detect_and_crop(mtcnn, new_image)#crop
    preprocessed_newface = pre_process(cropped_image)#preprocess
    preprocessed_newface_feature = run_model(interpreter, preprocessed_newface)

    # Load preprocessed_array
    preprocessed_images_directory = args.preprocessed_faces
    
    import math
    import os
    min_dist = math.inf
    min_i = math.inf
    i = 0
    file_path_min = ""
    for filename in os.listdir(preprocessed_images_directory):
        if filename.endswith(".npy"):
            file_path = os.path.join(preprocessed_images_directory, filename)
            feature_vector_file = np.load(file_path)
            feature_vector = run_model(interpreter, feature_vector_file)
            dist = euclidean_distance(feature_vector, preprocessed_newface_feature)
            if dist < min_dist:
                min_dist = dist
                min_i = i
                file_path_min = file_path
            i += 1
    
    print(min_dist)
    print(min_i)
    print(file_path_min)

if __name__ == '__main__':
    main()
