import pandas as pd
import os
from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv
from dash_user_interface import DashUserInterface
from simple_load_data import SimpleLoadData
from random_forest_loan_predictor import RandomForestLoanPredictor


# Main function
def main():
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    file_names = [
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

    container_name = 'blob-csv-files'
    download_path = "data/"

    data_loader = SimpleLoadData()

    data_loader.load(blob_service_client, container_name, file_names, download_path)

    train_data = pd.read_csv(f"{download_path}application_train.csv")

    model = RandomForestLoanPredictor()

    model.train(train_data)

    accuracy = model.evaluate()

    print(f"Accuracy: {accuracy}")

    user_interface = DashUserInterface()
    
    user_interface.display()

if __name__ == "__main__":
    load_dotenv()
    main()