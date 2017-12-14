from Recognizer import Recognizer
from Rectangle import Rectangle
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml

class FaceRecognizer(Recognizer):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def recognize(self, img):
        faces = self.detect_faces(img)

        rectangles = []
        for (x, y, width, height) in faces:
            rectangles.append(Rectangle(x, y, width, height))

        return rectangles

    def detect_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        return faces