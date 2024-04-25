## for ECE479 ICC Lab2 Part3

'''
*Main Student Script*
'''

# Your works start here

# Import packages you need here
from inception_resnet import InceptionResNetV1Norm
import numpy as np
import tensorflow as tf

# Create a model
model = InceptionResNetV1Norm()

# Verify the model and load the weights into the net
# print(model.summary())

# model.load_weights("./weights/inception_keras_weights.h5")  # Has been translated from checkpoint
model.load_weights("/mnt/c/Users/krish/Documents/UIUC/ECE479/lab3_sleeper_helper/Sleeper-helper/Facial Recognition/code/weights/inception_keras_weights.h5")
print(model.summary())
print("Number of layers:", len(model.layers))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
resnet_tflite_model = converter.convert()

with open('Resnet_model.tflite', 'wb') as f:
  f.write(resnet_tflite_model)
  f.close()