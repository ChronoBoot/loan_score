from abc import ABC, abstractmethod
from azure.storage.blob import BlobServiceClient
from typing import Any, List

class LoadData(ABC):
    """
    Abstract base class that defines a contract for loading data from Azure Blob Storage and saving it to a file.

    Any class that inherits from this must implement the `load` and `save` methods.
    """

    class LoadData(ABC):
        """
        Abstract base class that defines a contract for loading data from Azure Blob Storage and saving it to a file.

        Any class that inherits from this must implement the `load` and `save` methods.
        """

    @abstractmethod
    def load(self, file_names: List[str], download_path: str) -> None:
        """
        Load data from Azure Blob Storage.

        Args:
            file_names (List[str]): The list of file names to download.
            download_path (str): The local path to download the files.

        Returns:
            None
        """
        pass

    @abstractmethod
    def save(self, file_path: str, data: Any) -> None:
        """
        Save data to a file.

        Args:
            file_path (str): The path of the file to save.
            data (Any): The data to save.

        Returns:
            None
        """
        pass