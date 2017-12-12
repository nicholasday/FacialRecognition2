import numpy as np
import cv2
import os
import time
from numpy import linalg as LA
import requests as r
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml

cap = cv2.VideoCapture(0)

cv2.namedWindow('img')

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x-1, y-1), (x+w+1, y+h+1), (255, 0, 0), 2)
        
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == ord('`'):
        break
    
cap.release()
cv2.destroyAllWindows()
