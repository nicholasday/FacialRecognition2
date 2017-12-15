import cv2

class Rectangle:
    def __init__(self, x_init, y_init, width, height):
        self.x = x_init
        self.y = y_init
        self.width = width
        self.height = height

    def draw(self, img):
        cv2.rectangle(img, (self.x-1, self.y-1), (self.x+self.width+1, self.y+self.height+1), (255, 0, 0), 2)

    def __repr__(self):
        return 'Rectangle({self.x}, {self.y}, {self.width}, {self.height})'
