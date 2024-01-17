import os
import logging

from dotenv import load_dotenv
import requests
from backend.src.data_processing.load_data_abc import LoadData
import os
from backend.utils.profiling_utils import conditional_profile

class SimpleLoadData(LoadData):
    """
    A concrete implementation of the LoadData abstract base class.

    This class provides functionality to load data from Azure Blob Storage and save it to a local directory.
    The save method is not implemented and should be implemented in a subclass if needed.
    """

    CSV_URLS = [
        'https://loanscoringgit.blob.core.windows.net/blob-csv-files/application_test.csv',
        'https://loanscoringgit.blob.core.windows.net/blob-csv-files/application_train.csv',
        'https://loanscoringgit.blob.core.windows.net/blob-csv-files/bureau.csv',
        'https://loanscoringgit.blob.core.windows.net/blob-csv-files/bureau_balance.csv',
        'https://loanscoringgit.blob.core.windows.net/blob-csv-files/credit_card_balance.csv',
        'https://loanscoringgit.blob.core.windows.net/blob-csv-files/HomeCredit_columns_description.csv',
        'https://loanscoringgit.blob.core.windows.net/blob-csv-files/installments_payments.csv',
        'https://loanscoringgit.blob.core.windows.net/blob-csv-files/POS_CASH_balance.csv',
        'https://loanscoringgit.blob.core.windows.net/blob-csv-files/previous_application.csv',
    ]

    def __init__(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("SimpleLoadData initialized")

    @conditional_profile
    def download_file(self, url: str, filepath: str) -> None:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # This will check for any HTTP errors
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)

    @conditional_profile
    def load(self, file_urls: list, download_path: str) -> None:
        """
        Load data from Azure Blob Storage and save it to a local directory.

        Args:
            file_urls: The list of file urls to download.
            download_path: The local path to download the files.

        Returns:
            None
        """
        try:
            if not os.path.exists(download_path):
                os.makedirs(download_path)

            for url in file_urls:
                filename = url.split('/')[-1]  # Extracts the file name
                filepath = f"{download_path}/{filename}"
                self.download_file(url, filepath)
                logging.info(f"Downloaded file {filename} from Azure Blob Storage to {filepath}")
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