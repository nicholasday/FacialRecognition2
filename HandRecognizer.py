from Recognizer import Recognizer
from Rectangle import Rectangle
import numpy as np
import math
import cv2

backgroundModel = cv2.createBackgroundSubtractorMOG2()

class HandRecognizer(Recognizer):
    maskCreated = False
    mask = None
    maskIterations = 0

    def __init__(self, x_begin, y_end):
        self.x_begin = x_begin
        self.y_end = y_end

    def calculateFingers(self, res, drawing): 
        hull = cv2.convexHull(res, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(res, hull)
            cnt = 0
            if defects is not None:
                for i in range(defects.shape[0]):  # calculate the angle
                    s, e, f, d = defects[i][0]
                    start = tuple(res[s][0])
                    end = tuple(res[e][0])
                    far = tuple(res[f][0])
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                    if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                        cnt += 1
                        cv2.circle(drawing, far, 8, [211, 84, 0], -1)

                return cnt
        return 0

    def recognize(self, img):
        cv2.rectangle(img, (int(self.x_begin * img.shape[1]), 0), (img.shape[1], int(self.y_end * img.shape[0])), (255, 0, 0), 2)

        # if not self.maskCreated:
        # if self.maskCreated < 10:
            #self.mask = backgroundModel.apply(img)
            #kernel = np.ones((3, 3), np.uint8)
            #self.mask = cv2.erode(self.mask, kernel, iterations=1)
            #self.maskIterations = self.maskIterations + 1
            #self.maskCreated = True

        #res = cv2.bitwise_and(img, img, mask=self.mask)

        # roi = res[0:int(self.y_end * img.shape[0]), int(self.x_begin * img.shape[1]):img.shape[1]]
        # roi_img = img[0:int(self.y_end * img.shape[0]), int(self.x_begin * img.shape[1]):img.shape[1]]

        if not self.maskCreated:
            self.mask = img[0:int(self.y_end * img.shape[0]), int(self.x_begin * img.shape[1]):img.shape[1]]
            self.maskCreated = True
            return img
        
        roi = img[0:int(self.y_end * img.shape[0]), int(self.x_begin * img.shape[1]):img.shape[1]]
        roi = cv2.subtract(self.mask, roi)

        gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        ret,th2= cv2.threshold(gray_image,20,255,cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(th2, kernel, iterations=1)
        im2, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 300:
                hull = cv2.convexHull(contour)
                hull2 = cv2.convexHull(contour,returnPoints = False)
                defects = cv2.convexityDefects(contour,hull2)

                cv2.drawContours(roi, [hull], -1, 100, 3)
                cv2.drawContours(roi, [contour], -1, 255, 3)

                count = self.calculateFingers(contour, roi)

        img[0:int(self.y_end * img.shape[0]), int(self.x_begin * img.shape[1]):img.shape[1]] = roi # cv2.cvtColor(erosion, cv2.COLOR_GRAY2BGR)

        return img