import cv2

class Button:
    def __init__(self, text, topCorner, bottomCorner, color):
        self.text = text
        self.topCorner = topCorner
        self.bottomCorner = bottomCorner
        self.color = color

    def draw(self, img):
        cv2.rectangle(img, (self.topCorner.x, self.topCorner.y), (self.bottomCorner.x, self.bottomCorner.y), self.color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, self.text, (self.topCorner.x + 10, self.bottomCorner.y - 10), font, 2, (255, 255, 255), 2)

    def setClickHandler(self, handler):
        self.handler = handler

    def detectClick(self, x, y):
        if (x > self.topCorner.x and x < self.bottomCorner.x) and (y > self.topCorner.y and y < self.bottomCorner.y):
            self.handler()
