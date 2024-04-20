from fer import FER
import cv2
import pprint

img = cv2.imread("C:\\Users\\sravy\\OneDrive\\Desktop\\Personal projects\\Sleeper-helper\\web_interface\\uploads\\sravya2.jpeg")
detector = FER(mtcnn=True)

top_result, score = detector.top_emotion(img)
pprint.pprint("Top Result:", top_result, "\tScore:", score)
