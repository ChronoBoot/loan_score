from abc import ABC, abstractmethod

class LoadData(ABC):

    @abstractmethod
    def load(self, blob_service_client, container_name, file_names, download_path):
       pass

    @abstractmethod
    def save(self, file_path, data):
        pass