from fer import FER
import cv2
import pprint

img = cv2.imread("/mnt/c/Users/krish/Documents/UIUC/ECE479/lab3_sleeper_helper/Sleeper-helper/web_interface/uploads/neutral.jpg")
detector = FER()

result = detector.top_emotion(img)

print(result)

