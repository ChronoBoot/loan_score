"""
This script downloads a set of CSV files from Azure Blob Storage, 
loads them into pandas DataFrames, drops unnecessary columns, 
and uses a random forest model to make loan predictions.
"""

# Standard library imports
import os
import argparse
import logging
import traceback

# Related third party imports
import pandas as pd
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Local application/library specific imports
from src.ui.dash_user_interface import DashUserInterface
from src.data_processing.simple_load_data import SimpleLoadData
from src.models.random_forest_loan_predictor import RandomForestLoanPredictor

# Columns to drop from the dataset
COLUMNS_TO_DROP = ['SK_ID_CURR']  # ID column that's not needed for the analysis

# List of file names to download from Azure Blob Storage
FILE_NAMES = [
    'HomeCredit_columns_description.csv',
    'POS_CASH_balance.csv',
    'application_test.csv',
    'application_train.csv',
    'bureau.csv',
    'bureau_balance.csv',
    'credit_card_balance.csv',
    'installments_payments.csv',
    'previous_application.csv',
    'sample_submission.csv'
]

# Name of the container in Azure Blob Storage where the CSV files are stored
CONTAINER_NAME = 'blob-csv-files'

# Main function
def _main(FREQUENCY, DOWNLOAD_PATH):
    """
    This function is the entry point of the loan scoring application.
    
    Parameters:
    FREQUENCY (int): The frequency at which to sample the training data. Should not be modiefied.
    DOWNLOAD_PATH (str): The path to download the data from Azure Blob Storage. Should not be modified.
    
    Returns:
    None
    """
    try:
        # Get the Azure Storage connection string from environment variables
        connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        logging.info("Azure Storage connection string retrieved")

        # Load data from Azure Blob Storage
        data_loader = SimpleLoadData()
        data_loader.load(blob_service_client, CONTAINER_NAME, FILE_NAMES, DOWNLOAD_PATH)
        logging.info("Data loaded from Azure Blob Storage")

        # Read the training data
        train_data = pd.read_csv(f"{DOWNLOAD_PATH}application_train.csv")
        train_data = train_data.drop(columns=COLUMNS_TO_DROP)
        logging.info("Training data loaded and columns dropped")

        # Identify categorical and float columns
        # Is considered categorical if it's not float64 and a boolean represented with an integer
        categorical_columns = {
            col: train_data[col].dropna().unique().tolist()
            for col in train_data.columns
            if train_data[col].dtype != 'float64' and not (train_data[col].dtype == 'int64' and len(train_data[col].unique()) > 2)
        }
        logging.info("Categorical columns identified")

        # By definition the rest are float
        numerical_columns = [col for col in train_data.columns if col not in categorical_columns]
        logging.info("Numerical columns identified")

        categorical_columns.pop('TARGET', None)

        # Train the random forest model
        model = RandomForestLoanPredictor()
        model.train(train_data[::FREQUENCY])
        logging.info("Model trained")

        accuracy = model.evaluate()
        logging.info(f"Model evaluated with accuracy: {accuracy}")

        # Display the user interface
        user_interface = DashUserInterface(model, categorical_columns, numerical_columns)
        user_interface.display()
        logging.info("User interface displayed")

  
    except Exception as e:
        logging.error(f"An error occurred on line {traceback.extract_tb(e.__traceback__)[0].lineno}: {str(e)}")

# Entry point of the script
if __name__ == "__main__":
    try:
        load_dotenv()

        # Configure logging
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Loan prediction application")
        parser.add_argument('--frequency', type=int, default=10, help='The sampling frequency for the data. If set to 10, every 10th line from the CSV files will be used.')        
        parser.add_argument('--download_path', type=str, default='data/', help='The directory where downloaded files will be stored. If the directory does not exist, it will be created.')
        args = parser.parse_args()

        # Validate command line arguments
        if args.frequency <= 0:
            raise argparse.ArgumentTypeError("Frequency must be a positive integer.")
        
        # Normalize the path (remove redundant separators and up-level references)
        normalized_path = os.path.normpath(args.download_path)

        # Check if the path is a directory
        if not os.path.isdir(normalized_path):
            try:
                # Try to create the directory
                os.makedirs(normalized_path)
            except OSError:
                logging.error(f"Invalid path: {args.download_path}")
        else:
            logging.info(f"Directory {normalized_path} already exists")

        # Extract the frequency and download path from command line arguments
        FREQUENCY = args.frequency
        DOWNLOAD_PATH = args.download_path

        # Call the main function
        _main(FREQUENCY, DOWNLOAD_PATH)

    except argparse.ArgumentError as e:
        logging.error(f"Invalid command line argument: {str(e)}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}\n{traceback.format_exc()}")