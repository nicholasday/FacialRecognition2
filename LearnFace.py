import os
import cv2
import numpy as np

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
