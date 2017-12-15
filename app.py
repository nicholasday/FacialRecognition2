import cv2
from Trainer import Trainer
from Button import Button
from Rectangle import Rectangle
from Point import Point
from FaceRecognizer import FaceRecognizer
from HandRecognizer import HandRecognizer
from Database import Database
import numpy as np

cap = cv2.VideoCapture(0)

cv2.namedWindow('img')

trainer = Trainer()
faceRecognizer = FaceRecognizer()
handRecognizer = HandRecognizer(.65, .8)
db = Database()

x = 0
y = 0

def clickHandler(event, x_click, y_click, flags, param):
    global x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        x = x_click
        y = y_click
    elif event == cv2.EVENT_LBUTTONUP:
        x = 0
        y = 0

cv2.setMouseCallback("img", clickHandler)

buttons = []

listButton = Button("List", Point(100, 100), Point(200, 200), (0, 255, 0))
listButton.setClickHandler(db.getMembers)
buttons.append(listButton)

learnButton = Button('Learn', Point(250, 100), Point(350, 200), (0, 255, 0))
learnButton.setClickHandler(trainer.learnFace)
buttons.append(learnButton)

while 1:
    ret, img = cap.read()

    img = cv2.flip(img, 1)
    
    faces = faceRecognizer.detect_faces(img)

    img = handRecognizer.recognize(img)

    for face in faces:
        face.draw(img)

    img = faceRecognizer.recognize(img)

    for button in buttons:
        button.draw(img)
        if isinstance(button, Button):
            button.detectClick(x, y)

    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == ord('`'):
        break
    
cap.release()
cv2.destroyAllWindows()
