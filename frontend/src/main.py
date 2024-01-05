"""
This script downloads a set of CSV files from Azure Blob Storage, 
loads them into pandas DataFrames, drops unnecessary columns, 
and uses a random forest model to make loan predictions.
"""

# Standard library imports
import json
import os
import argparse
import logging
import traceback

# Related third party imports
import pandas as pd
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Local application/library specific imports
from frontend.src.ui.dash_user_interface import DashUserInterface
import requests

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

API_URL = "http://127.0.0.1:5000"
TRAIN_URL = f"{API_URL}/train"
PREDICT_URL = f"{API_URL}/predict"
EVALUATE_URL = f"{API_URL}/evaluate"
MOST_IMPORTANT_FEATURES_URL = f"{API_URL}/most_important_features"

# Get the absolute path of the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COLUMNS_INFO_FILENAME = "columns_info.json"
COLUMNS_INFO_PATH = os.path.join(SCRIPT_DIR, os.pardir, os.pardir, 'shared_config', COLUMNS_INFO_FILENAME)

def get_categorical_columns(columns_info : dict) -> dict:
    categorical_columns = {}

    for table in columns_info:
        for column in columns_info[table]:
            if column['values'] != None:
                categorical_columns[column['fieldname']] = {
                    key: value for key, value in column.items() if key != 'fieldname'
                }

    return categorical_columns

def get_numerical_columns(columns_info: dict) -> dict:
    numerical_columns = {}

    for table in columns_info:
        for column in columns_info[table]:
            if column['values'] == None:
                numerical_columns[column['fieldname']] = {
                    key: value for key, value in column.items() if key != 'fieldname'
                }
    
    return numerical_columns 

# Main function
def _main(FREQUENCY : int):
    """
    This function is the entry point of the loan scoring application.
    
    Parameters:
    FREQUENCY (int): The frequency at which to sample the training data. Should not be modiefied.
    
    Returns:
    None
    """
    try:

        # Read the JSON file
        with open(COLUMNS_INFO_PATH, 'r') as file:
            columns_info = json.load(file)['tables']

        categorical_columns = get_categorical_columns(columns_info)
        logging.info("Categorical columns identified")
        
        numerical_columns = get_numerical_columns(columns_info)
        logging.info("Numerical columns identified")

        categorical_columns.pop('TARGET', None)

        # Train the model
        data = {
            "sampling_frequency": FREQUENCY,
            "target_variable": "TARGET",
            "concat": "False"
        }

        response = requests.post(TRAIN_URL, json=data)
        if(response.status_code != 200):
            raise Exception(f"An error occurred while training the model: {response.json()['message']}")
        
        logging.info(f"Model trained successfully: {response.json()['message']}")

        # Evaluate the model
        response = requests.get(EVALUATE_URL)
        if(response.status_code != 200):
            raise Exception(f"An error occurred while evaluating the model: {response.json()['message']}")
        
        response_json = response.json()
        accuracy = response_json['accuracy']
        logging.info(f"Model evaluated with accuracy: {accuracy}")

        # Display the most important features
        response = requests.post(MOST_IMPORTANT_FEATURES_URL, json={"nb_features": 10})
        if(response.status_code != 200):
            raise Exception(f"An error occurred while getting the most important features: {response.json()['message']}")
        
        response_json = response.json()
        most_important_features = response_json['features']
        logging.info(f"Most important features:\n{most_important_features}")


        # Display the user interface
        user_interface = DashUserInterface(categorical_columns, numerical_columns)
        logging.info("User interface displayed")
        user_interface.display()

       
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
        args = parser.parse_args()

        # Validate command line arguments
        if args.frequency <= 0:
            raise argparse.ArgumentTypeError("Frequency must be a positive integer.")
        
        # Extract the frequency and download path from command line arguments
        FREQUENCY = args.frequency

        # Call the main function
        _main(FREQUENCY)

    except argparse.ArgumentError as e:
        logging.error(f"Invalid command line argument: {str(e)}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}\n{traceback.format_exc()}")