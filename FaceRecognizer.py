from numpy import linalg as LA
import os
import numpy as np
from Recognizer import Recognizer
from Rectangle import Rectangle
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml

class FaceRecognizer(Recognizer):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def recognize(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x - 1, y - 1), (x + w + 1, y + h + 1), (255, 0, 0), 2)
            crop_img = gray[y:(y + h), x:(x + w)]
            resized_crop_img = cv2.resize(crop_img, (50, 50))
            givenImage = resized_crop_img
            eigenvectors = np.load('eigenVectors.npy')
            average = cv2.imread('averageFace.png', cv2.IMREAD_GRAYSCALE)
            resized_crop_img = resized_crop_img.astype(np.int16)
            average = average.astype(np.int16)
            resized_crop_img = resized_crop_img - average
            resized_crop_img = resized_crop_img.flatten()
            resized_crop_img = np.dot(resized_crop_img, eigenvectors)
            distances = []
            peopleNames = []
            for person in os.listdir('learnedFaces'):
                learned = np.load('learnedFaces/' + person)
                peopleNames.append(person[:-5])
                diff = resized_crop_img - learned
                distance = LA.norm(diff)
                distances.append(distance)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, peopleNames[distances.index(min(distances))], (50, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        return img

    def detect_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        rectangles = []
        for (x, y, width, height) in faces:
            rectangles.append(Rectangle(x, y, width, height))

        return rectangles
