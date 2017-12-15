from abc import ABC, abstractmethod

class Recognizer(ABC):
    @abstractmethod
    def recognize(self):
        pass
