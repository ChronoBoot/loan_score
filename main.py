import pandas as pd
import os
from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv

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

    container_client = blob_service_client.get_container_client(container=container_name)

    # Verify if download_path directory exists, otherwise create it
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    for blob_name in file_names:
        download_file_path = f"{download_path}{blob_name}"
        if not os.path.exists(download_file_path):
            blob_client = container_client.get_blob_client(blob_name)
            with blob_client.download_blob() as blob:
                with open(download_file_path, "wb") as download_file:
                    chunk_size = 1024 * 1024  # 1MB
                    offset = 0
                    while offset < blob.size:
                        download_file.write(blob.read(offset=offset, length=chunk_size))
                        offset += chunk_size

if __name__ == "__main__":
    load_dotenv()
    main()