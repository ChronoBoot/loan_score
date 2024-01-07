from abc import ABC, abstractmethod
import pandas as pd

class ReadDataABC(ABC):
    """
    Abstract base class for reading data.

    This class serves as a blueprint for any class that needs to implement a method for reading data.
    """

    POS_CASH_BALANCE_NAME = 'POS_CASH_balance.csv'
    APPLICATION_TEST_NAME = 'application_test.csv'
    APPLICATION_TRAIN_NAME = 'application_train.csv'
    BUREAU_NAME = 'bureau.csv'
    BUREAU_BALANCE_NAME = 'bureau_balance.csv'
    CREDIT_CARD_BALANCE_NAME = 'credit_card_balance.csv'
    INSTALLMENTS_PAYMENTS_NAME = 'installments_payments.csv'
    PREVIOUS_APPLICATION_NAME = 'previous_application.csv'
    FILES_NAMES = [POS_CASH_BALANCE_NAME, APPLICATION_TEST_NAME, APPLICATION_TRAIN_NAME, BUREAU_NAME, BUREAU_BALANCE_NAME, CREDIT_CARD_BALANCE_NAME, INSTALLMENTS_PAYMENTS_NAME, PREVIOUS_APPLICATION_NAME]

    @abstractmethod
    def retrieve_data(self, files_path: str, concat: bool, sampling_frequency: int) -> pd.DataFrame:
        """
        Abstract method for reading data.

        This method should be implemented by any concrete class that inherits from this abstract base class.

        Parameters:
        files_path (str): The path where the files are located.
        concat (bool): Whether to read and concatenate the data from all the files or just read the main file.
        sampling_frequency (int): The sampling frequency to use when reading the data. 10 means 1 out of 10 rows will be read.

        Returns:
        pd.DataFrame: The data read from the files as a pandas DataFrame.
        """
        pass