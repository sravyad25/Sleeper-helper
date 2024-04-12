import argparse
import time
import numpy as np
import tflite_runtime.interpreter as tflite
from pycoral.utils.edgetpu import make_interpreter

def main(args):
    # interpreter = tflite.Interpreter(args.model) # if using CPU
    interpreter = make_interpreter(args.model) # if using TPU
    interpreter.allocate_tensors()
    
    train_images = np.load("/home/admin/fashion_train_images.npy")
    test_images = np.load("/home/admin/fashion_test_images.npy")
    train_labels = np.load("/home/admin/fashion_train_labels.npy")
    test_labels = np.load("/home/admin/fashion_test_labels.npy")

    test_images = test_images.astype(np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct_predictions = 0
    total_samples = len(test_images)
    start_time = time.time()
    for i in range(total_samples):
        interpreter.set_tensor(input_details[0]['index'], test_images[i:i+1].astype(np.float32))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_label = np.argmax(output_data)
        true_label = test_labels[i]
        if predicted_label == true_label:
            correct_predictions += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    accuracy = correct_predictions / total_samples
    print("Model Output:", output_data)
    print("Model Accuracy:", accuracy)
    print("Model Time:", elapsed_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fashion MNIST Inference')
    parser.add_argument('-m', '--model', required=True, help='Path to the TFLite model file')
    args = parser.parse_args()
    main(args)

