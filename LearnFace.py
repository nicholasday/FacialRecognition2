from Trainer import Trainer
from FaceRecognizer import FaceRecognizer
import os
import cv2
import numpy as np

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml

trainer = Trainer()
faceRecognizer = FaceRecognizer()
cap = cv2.VideoCapture(0)
learned = False

while 1:
    ret, img = cap.read()

    img = cv2.flip(img, 1)
    
    faces = faceRecognizer.detect_faces(img)

    for face in faces:
        face.draw(img)

    if not faceRecognizer.pictureTaken:
        img = faceRecognizer.takePictures(img, 5)

    if faceRecognizer.pictureTaken and not learned:
        img=trainer.learnGiven(img)
        learned = True

    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == ord('`'):
        break
    
cap.release()
cv2.destroyAllWindows()

