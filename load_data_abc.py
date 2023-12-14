from abc import ABC, abstractmethod

class LoadData(ABC):

    @abstractmethod
    def load(self, file_path):
        pass

    @abstractmethod
    def save(self, file_path, data):
        pass