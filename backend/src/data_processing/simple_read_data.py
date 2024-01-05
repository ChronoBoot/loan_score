import pandas as pd
from backend.src.data_processing.read_data_abc import ReadDataABC

class SimpleReadData(ReadDataABC):
    """
    Simple implementation of the ReadDataABC abstract base class.

    This class provides a simple way to read data from CSV files.
    """

    def read_data(self, files_path: str, concat: bool, sampling_frequency: int) -> pd.DataFrame:
        """
        Read data from a list of CSV files.

        This method reads each file in files_names from the directory specified by files_path.
        It assumes that the files are in CSV format.

        Parameters:
        files_path (str): The path where the files are located.
        concat (bool): Whether to read and concatenate the data from all the files or just read the main file.
        sampling_frequency (int): The sampling frequency to use when reading the data. 10 means 1 out of 10 rows will be read.

        Returns:
        pd.DataFrame: The data read from the files as a pandas DataFrame.
        """
        # Initialize an empty list to store the data from each file
        data = []

        data = pd.read_csv(f"{files_path}/{ReadDataABC.APPLICATION_TRAIN_NAME}", skiprows=lambda x: x % sampling_frequency != 0)

        if concat:
            raise NotImplementedError
            
        # Concatenate all the data into a single DataFrame
        return data