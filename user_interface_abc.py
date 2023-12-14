from abc import ABC, abstractmethod

class UserInterface(ABC):
    
    @abstractmethod
    def display(self):
        pass

    @abstractmethod
    def send_data(self):
        pass