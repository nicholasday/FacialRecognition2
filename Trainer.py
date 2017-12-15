import numpy as np
import cv2
import os

class Trainer():
    def train(self):
        z = 2
        numberOfImages = 0
        for images in os.listdir('trainingImages') :
            numberOfImages+=1
        im = cv2.imread('trainingImages/myFace0.png', cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread('trainingImages/myFace1.png', cv2.IMREAD_GRAYSCALE)
        im = im.flatten()
        im2 = im2.flatten()
        m = np.column_stack((im, im2))
        while z < numberOfImages :
            img = cv2.imread('trainingImages/myFace' + str(z) + '.png', cv2.IMREAD_GRAYSCALE)
            img = img.flatten()
            m = np.concatenate((m, img[:, None]), axis=1)
            z += 1
        np.save('trainingFacesMatrix.npy', m)

    def learnFace(self):
