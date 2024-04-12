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

def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m', '--model', required=True, help='File path of .tflite file.')
    parser.add_argument(
        '-s1', '--sravya1', required=True, help='File path of Sravya 1 image.')
    parser.add_argument(
        '-s2', '--sravya2', required=True, help='File path of Sravya 2 image.')
    parser.add_argument(
        '-k1', '--krish1', required=True, help='File path of Krish 1 image.')
    parser.add_argument(
        '-k2', '--krish2', required=True, help='File path of Krish 2 image.')
    args = parser.parse_args()

    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    # Load images
    image_sravya_1 = np.load(args.sravya1)
    image_sravya_2 = np.load(args.sravya2) 
    image_krish_1 = np.load(args.krish1) 
    image_krish_2 = np.load(args.krish2)

    # Run inference for each image
    feature_vector_sravya1 = run_model(interpreter, image_sravya_1)
    feature_vector_sravya2 = run_model(interpreter, image_sravya_2)
    feature_vector_krish1 = run_model(interpreter, image_krish_1)
    feature_vector_krish2 = run_model(interpreter, image_krish_2)

    # Calculate Euclidean distances
    distance_sravya1_sravya2 = euclidean_distance(feature_vector_sravya1, feature_vector_sravya2)
    distance_sravya1_krish1 = euclidean_distance(feature_vector_sravya1, feature_vector_krish1)
    distance_krish1_krish2 = euclidean_distance(feature_vector_krish1, feature_vector_krish2)
    distance_krish1_sravya2 = euclidean_distance(feature_vector_krish1, feature_vector_sravya2)
    distance_sravya1_krish2 = euclidean_distance(feature_vector_sravya1, feature_vector_krish2)
    distance_krish2_sravya2 = euclidean_distance(feature_vector_krish2, feature_vector_sravya2)

    print("Distance between Sravya 1 and Sravya 2:", distance_sravya1_sravya2)
    print("Distance between Sravya 1 and Krish 1:", distance_sravya1_krish1)
    print("Distance between Krish 1 and Krish 2:", distance_krish1_krish2)
    print("Distance between Krish 1 and Sravya 2:", distance_krish1_sravya2)
    print("Distance between Sravya 1 and Krish 2:", distance_sravya1_krish2)
    print("Distance between Krish 2 and Sravya 2:", distance_krish2_sravya2)

if __name__ == '__main__':
    main()
