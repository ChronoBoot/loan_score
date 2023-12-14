from abc import ABC, abstractmethod

class LoanPredictor(ABC):

    @abstractmethod
    def predict(self, loan):
        pass

    @abstractmethod
    def train(self, loans):
        pass