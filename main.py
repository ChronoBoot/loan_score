import io
import pandas as pd
import os
from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv
import logging
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Function to download files from Azure Blob Storage
def download_files(blob_service_client, container_name, file_names, download_path):
    container_client = blob_service_client.get_container_client(container=container_name)

    if not os.path.exists(download_path):
        os.makedirs(download_path)

    for blob_name in file_names:
        download_file_path = f"{download_path}{blob_name}"
        if not os.path.exists(download_file_path):
            blob_client = container_client.get_blob_client(blob_name)
            with open(download_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            logging.info(f"Downloaded file: {blob_name}")

# Function to preprocess the training data
def preprocess_data(train_data, target_variable):
    # Drop the target variable from the training data
    X = train_data.drop(columns=[target_variable])
    
    # Initialize a LabelEncoder
    le = LabelEncoder()

    # Apply the LabelEncoder to each column
    X = X.apply(lambda col: le.fit_transform(col) if col.dtype == 'object' else col)

    # Fill NaN values
    X = X.fillna(0)

    y = train_data[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to train the machine learning model
def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Function to calculate the accuracy of the model
def calculate_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

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

    download_files(blob_service_client, container_name, file_names, download_path)

    train_data = pd.read_csv(f"{download_path}application_train.csv")

    target_variable = 'TARGET'

    X_train, X_test, y_train, y_test = preprocess_data(train_data, target_variable)

    model = train_model(X_train, y_train)

    accuracy = calculate_accuracy(model, X_test, y_test)
    print(f"Accuracy: {accuracy}")

    # Save the model
    joblib.dump(model, 'model.pkl')

    if __name__ == "__main__":
        load_dotenv()
        logging.basicConfig(level=logging.INFO)
        main()

    accuracy = calculate_accuracy(model, X_test, y_test)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    main()