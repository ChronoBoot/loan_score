import os
import logging

from dotenv import load_dotenv
from backend.src.data_processing.load_data_abc import LoadData
from azure.storage.blob import BlobServiceClient

class SimpleLoadData(LoadData):
    """
    A concrete implementation of the LoadData abstract base class.

    This class provides functionality to load data from Azure Blob Storage and save it to a local directory.
    The save method is not implemented and should be implemented in a subclass if needed.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of the SimpleLoadData class.

        The SimpleLoadData class provides functionality to load data from Azure Blob Storage and save it to a local directory.

        Args:
            None

        Returns:
            None
        """

        load_dotenv()
        self.blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))
        self.container_name = os.getenv('AZURE_STORAGE_CONTAINER_NAME')

    def load(self, file_names : list, download_path: str):
        """
        Load data from Azure Blob Storage and save it to a local directory.

        Args:
            blob_service_client: The BlobServiceClient instance.
            container_name: The name of the container.
            file_names: The list of file names to download.
            download_path: The local path to download the files.

        Returns:
            None
        """
        try:
            container_client = self.blob_service_client.get_container_client(container=self.container_name)

            if not os.path.exists(download_path):
                os.makedirs(download_path)

            for blob_name in file_names:
                download_file_path = f"{download_path}/{blob_name}"
                if not os.path.exists(download_file_path):
                    blob_client = container_client.get_blob_client(blob_name)
                    with open(download_file_path, "wb") as download_file:
                        download_file.write(blob_client.download_blob().readall())
        except Exception as e:
            logging.error(f"Failed to load data from Azure Blob Storage: {e}")

    def save(self, file_path, data):
        """
        Save data to a file. This method is not implemented in this class and should be implemented in a subclass if needed.

        Args:
            file_path: The path of the file to save.
            data: The data to save.

        Raises:
            NotImplementedError: The save method is not implemented in SimpleLoadData and should be implemented in a subclass if needed.

        Returns:
            None
        """
        raise NotImplementedError("The save method is not implemented in SimpleLoadData and should be implemented in a subclass if needed.")