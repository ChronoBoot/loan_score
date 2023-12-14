from abc import ABC, abstractmethod

class UserInterface(ABC):
    
    @abstractmethod
    def display(self):
        pass