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

    def learnGiven(self, img):
        cv2.putText(img, 'Type name of person being learned into console.', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        numOfSaves = 0
        name = input('input name of person being learned')
        names = []
        matches = 0
        for person in os.listdir('learnedFaces'):
            learned = np.load('learnedFaces/' + person)
            names.append(person[:-5])
        print(names)
        for person in os.listdir('learnedFaces'):
            if name == person[:-5]:
                matches += 1
                print('match')
        tempName = name
        tempName += str(matches)
        eigenvectors = np.load('eigenVectors.npy')
        average = cv2.imread('averageFace.png', cv2.IMREAD_GRAYSCALE)
        given = cv2.imread('givenFace' + str(numOfSaves) + '.png', cv2.IMREAD_GRAYSCALE)
        given = given.astype(np.int16)
        average = average.astype(np.int16)
        given = given - average
        given = given.flatten()
        learned = np.dot(given, eigenvectors)
        np.save('learnedFaces/' + tempName + '.npy', learned)
        return img
