import json
import numpy as np
import pandas as pd
from backend.src.data_processing.read_data_abc import ReadDataABC

class SimpleReadData(ReadDataABC):
    """
    Simple implementation of the ReadDataABC abstract base class.

    This class provides a simple way to read data from CSV files.
    """

    def one_hot_encode(self, data: pd.DataFrame, columns_names: list) -> pd.DataFrame:
        """
        One-hot encode the categorical columns in the data.

        Parameters:
        data (pd.DataFrame): The data to one-hot encode.
        columns_names (str): The names of the columns to one-hot encode.

        Returns:
        pd.DataFrame: The one-hot encoded data concated with the original data.
        """
        prefixes = [column_name for column_name in columns_names]

        one_hot_encoded_data = pd.concat([data] + [pd.get_dummies(data[col], prefix=pre) for col, pre in zip(columns_names, prefixes)], axis=1)

        return one_hot_encoded_data
    
    def update_aggregation_dict(self, data: pd.DataFrame, aggregation_dict: dict, prefixes: list) -> None:
        """
        Update the aggregation dictionary with a count for each column starting with the prefixes.

        Parameters:
        aggregation_dict (dict): The aggregation dictionary to update.
        prefixes (list): The prefixes to add to the aggregation dictionary.

        """
        for prefix in prefixes:
            aggregation_dict.update({
                col: ['sum'] for col in data.columns if col.startswith(f"{prefix}_")
            })

    def flatten_and_reset_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten the MultiIndex columns and reset the index.

        Parameters:
        data (pd.DataFrame): The data to flatten and reset.
        """
        # Flatten the MultiIndex columns
        data.columns = ['_'.join(col).strip() for col in data.columns.values]

        # Reset the index
        data.reset_index(inplace=True)

    def aggregate_data(self, data: pd.DataFrame, aggregation_dict: dict, prefixes: list, groupby_col: str) -> pd.DataFrame:
        """
        Aggregate the data.

        Parameters:
        data (pd.DataFrame): The data to aggregate.
        aggregation_dict (dict): The aggregation dictionary to use.
        prefixes (list): The prefixes to add to the aggregation dictionary.
        groupby_col (str): The column to group by.

        Returns:
        pd.DataFrame: The aggregated data.
        """
        self.update_aggregation_dict(data, aggregation_dict, prefixes)

        aggregated_data = data.groupby(groupby_col).agg(aggregation_dict)

        self.flatten_and_reset_index(aggregated_data)

        return aggregated_data

    def get_aggregated_bureau_data(self, bureau_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate the data from bureau_balance.

        Parameters:
        bureau_data (pd.DataFrame): The data from the bureau_balance table.

        Returns:
        pd.DataFrame: The aggregated data. Grouped by SK_ID_CURR. 
        """

        # Calculate the mean difference between previous credits
        bureau_data_days_credit = bureau_data['DAYS_CREDIT'].sort_values()
        bureau_data_days_credit_diff = bureau_data_days_credit.diff()
        bureau_data_days_credit_diff_mean = bureau_data_days_credit_diff.mean()            

        # Calculte the duration of the credit
        # We either use the factual end date or the expected end date (if credit is not closed)
        bureau_data = bureau_data.assign(
            CREDIT_DURATION = np.where(
                bureau_data['DAYS_ENDDATE_FACT'].isnull(),
                bureau_data['DAYS_CREDIT_ENDDATE'],
                bureau_data['DAYS_ENDDATE_FACT']
            )
            - bureau_data['DAYS_CREDIT'] 
        )
        bureau_data['CREDIT_DURATION'] = bureau_data['CREDIT_DURATION'].fillna(0).astype('int64')

        one_hot_encoding_columns = [
            'CREDIT_ACTIVE',
            'CREDIT_CURRENCY',
            'CREDIT_TYPE'
        ]

        bureau_data = self.one_hot_encode(bureau_data, one_hot_encoding_columns)

        aggregation_dict = {
            'DAYS_CREDIT': ['max', 'min'],
            'CREDIT_DAY_OVERDUE': ['max', 'min', 'mean'],
            'CREDIT_DURATION': ['max', 'min', 'mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['max', 'min', 'mean'],
            'CNT_CREDIT_PROLONG': ['max', 'min', 'mean'],
            'AMT_CREDIT_SUM': ['max', 'min', 'mean'],
            'AMT_CREDIT_SUM_DEBT': ['max', 'min', 'mean'],
            'AMT_CREDIT_SUM_LIMIT': ['max', 'min', 'mean'],
            'AMT_CREDIT_SUM_OVERDUE': ['max', 'min', 'mean'],
            'DAYS_CREDIT_UPDATE': ['min'],
            'AMT_ANNUITY': ['max', 'min', 'mean'],
        }

        aggregated_bureau_data = self.aggregate_data(bureau_data, aggregation_dict, one_hot_encoding_columns, 'SK_ID_CURR')
        aggregated_bureau_data['DAYS_CREDIT_DIFF_MEAN'] = bureau_data_days_credit_diff_mean

        return aggregated_bureau_data
    
    def get_aggregated_credit_card_balance_data(self, credit_card_balance_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate the data from credit_card_balance.

        Parameters:
        credit_card_balance_data (pd.DataFrame): The data from the credit_card_balance table.

        Returns:
        pd.DataFrame: The aggregated data. Grouped by SK_ID_CURR.
        """

        # One-hot encoding on NAME_CONTRACT_STATUS
        credit_card_balance_data = self.one_hot_encode(credit_card_balance_data, ['NAME_CONTRACT_STATUS'])
    
        aggregate_dict = {
            'MONTHS_BALANCE': ['max', 'min'],
            'AMT_BALANCE': ['max', 'min', 'mean'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'min', 'mean'],
            'AMT_DRAWINGS_ATM_CURRENT': ['max', 'min', 'mean'],
            'AMT_DRAWINGS_CURRENT': ['max', 'min', 'mean'],
            'AMT_DRAWINGS_OTHER_CURRENT': ['max', 'min', 'mean'],
            'AMT_DRAWINGS_POS_CURRENT': ['max', 'min', 'mean'],
            'AMT_INST_MIN_REGULARITY': ['max', 'min', 'mean'],
            'AMT_PAYMENT_CURRENT': ['max', 'min', 'mean'],
            'AMT_PAYMENT_TOTAL_CURRENT': ['max', 'min', 'mean'],
            'AMT_RECEIVABLE_PRINCIPAL': ['max', 'min', 'mean'],
            'AMT_RECIVABLE': ['max', 'min', 'mean'],
            'AMT_TOTAL_RECEIVABLE': ['max', 'min', 'mean'],
            'CNT_DRAWINGS_ATM_CURRENT': ['max', 'min', 'mean'],
            'CNT_DRAWINGS_CURRENT': ['max', 'min', 'mean'],
            'CNT_DRAWINGS_OTHER_CURRENT': ['max', 'min', 'mean'],
            'CNT_DRAWINGS_POS_CURRENT': ['max', 'min', 'mean'],
            'CNT_INSTALMENT_MATURE_CUM': ['max', 'min', 'mean'],
            'SK_DPD': ['max', 'min', 'mean'],
            'SK_DPD_DEF': ['max', 'min', 'mean']
        }

        aggregated_credit_card_balance_data = self.aggregate_data(credit_card_balance_data, aggregate_dict, ['NAME_CONTRACT_STATUS'], 'SK_ID_CURR')

        return aggregated_credit_card_balance_data

    def get_aggregated_installments_payments_data(self, installments_payments_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate the data from installments_payments.

        Parameters:
        installments_payments_data (pd.DataFrame): The data from the installments_payments table.

        Returns:
        pd.DataFrame: The aggregated data. Grouped by SK_ID_CURR.
        """

        aggregate_dict = {
            'NUM_INSTALMENT_VERSION': ['max', 'min', 'mean'],
            'NUM_INSTALMENT_NUMBER': ['max', 'min', 'mean'],
            'DAYS_INSTALMENT': ['max', 'min'],
            'DAYS_ENTRY_PAYMENT': ['max', 'min'],
            'AMT_INSTALMENT': ['max', 'min', 'mean'],
            'AMT_PAYMENT': ['max', 'min', 'mean']
        }

        aggregated_installments_payments_data = self.aggregate_data(installments_payments_data, aggregate_dict, [], 'SK_ID_CURR')

        return aggregated_installments_payments_data
    
    def get_aggregated_previous_application_data(self, previous_application_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate the data from previous_application.

        Parameters:
        previous_application_data (pd.DataFrame): The data from the previous_application table.

        Returns:
        pd.DataFrame: The aggregated data. Grouped by SK_ID_CURR.
        """

        # One-hot encoding of categorical columns
        categorical_columns = [
            'NAME_CONTRACT_TYPE',
            'WEEKDAY_APPR_PROCESS_START',
            'FLAG_LAST_APPL_PER_CONTRACT',
            'NFLAG_LAST_APPL_IN_DAY',
            'NAME_CASH_LOAN_PURPOSE',
            'NAME_CONTRACT_STATUS',
            'NAME_PAYMENT_TYPE',
            'CODE_REJECT_REASON',
            'NAME_TYPE_SUITE',
            'NAME_CLIENT_TYPE',
            'NAME_GOODS_CATEGORY',
            'NAME_PORTFOLIO',
            'NAME_PRODUCT_TYPE',
            'CHANNEL_TYPE',
            'NAME_SELLER_INDUSTRY',
            'NAME_YIELD_GROUP',
            'PRODUCT_COMBINATION',
            'NFLAG_INSURED_ON_APPROVAL'
        ]

        previous_application_data['NFLAG_INSURED_ON_APPROVAL'] = previous_application_data['NFLAG_INSURED_ON_APPROVAL'].fillna(0).astype('int64')
        previous_application_data = self.one_hot_encode(previous_application_data, categorical_columns)
        
        aggregate_dict = {
            'AMT_ANNUITY': ['max', 'min', 'mean'],
            'AMT_APPLICATION': ['max', 'min', 'mean'],
            'AMT_CREDIT': ['max', 'min', 'mean'],
            'AMT_DOWN_PAYMENT': ['max', 'min', 'mean'],
            'AMT_GOODS_PRICE': ['max', 'min', 'mean'],
            'HOUR_APPR_PROCESS_START': ['max', 'min', 'mean'],
            'RATE_DOWN_PAYMENT': ['max', 'min', 'mean'],
            'RATE_INTEREST_PRIMARY': ['max', 'min', 'mean'],
            'RATE_INTEREST_PRIVILEGED': ['max', 'min', 'mean'],
            'DAYS_DECISION': ['max', 'min', 'mean'],
            'SELLERPLACE_AREA': ['max', 'min', 'mean'],
            'CNT_PAYMENT': ['max', 'min', 'mean'],
            'DAYS_FIRST_DRAWING': ['max', 'min', 'mean'],
            'DAYS_FIRST_DUE': ['max', 'min', 'mean'],
            'DAYS_LAST_DUE_1ST_VERSION': ['max', 'min', 'mean'],
            'DAYS_LAST_DUE': ['max', 'min', 'mean'],
            'DAYS_TERMINATION': ['max', 'min', 'mean'],
        }

        aggregated_previous_application_data = self.aggregate_data(previous_application_data, aggregate_dict, categorical_columns, 'SK_ID_CURR')

        return aggregated_previous_application_data

    def get_aggregated_pos_cash_balance_data(self, pos_cash_balance_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate the data from pos_cash_balance.

        Parameters:
        pos_cash_balance_data (pd.DataFrame): The data from the pos_cash_balance table.

        Returns:
        pd.DataFrame: The aggregated data. Grouped by SK_ID_CURR.
        """
        
        # One-hot encoding of categorical columns
        pos_cash_balance_data = self.one_hot_encode(pos_cash_balance_data, ['NAME_CONTRACT_STATUS'])

        aggregate_dict = {
            'MONTHS_BALANCE': ['max', 'min'],
            'CNT_INSTALMENT': ['max', 'min', 'mean'],
            'CNT_INSTALMENT_FUTURE': ['max', 'min', 'mean'],
            'SK_DPD': ['max', 'min', 'mean'],
            'SK_DPD_DEF': ['max', 'min', 'mean']
        }

        aggregated_pos_cash_balance_data = self.aggregate_data(pos_cash_balance_data, aggregate_dict, ['NAME_CONTRACT_STATUS'], 'SK_ID_CURR')

        return aggregated_pos_cash_balance_data
    
    def retrieve_data(self, files_path: str, sampling_frequency: int, training : bool = True) -> pd.DataFrame:
        """
        Read data from a list of CSV files.

        This method reads each file in files_names from the directory specified by files_path.
        It assumes that the files are in CSV format.

        Parameters:
        files_path (str): The path where the files are located.
        sampling_frequency (int): The sampling frequency to use when reading the data. 10 means 1 out of 10 rows will be read.

        Returns:
        pd.DataFrame: The data read from the files as a pandas DataFrame.
        """
        # Initialize an empty list to store the data from each file
        data = []

        if training:
            train_data = pd.read_csv(f"{files_path}/{ReadDataABC.APPLICATION_TRAIN_NAME}", skiprows=lambda i: i % sampling_frequency != 0)
            data = train_data
        else:
            test_data = pd.read_csv(f"{files_path}/{ReadDataABC.APPLICATION_TEST_NAME}", skiprows=lambda i: i % sampling_frequency != 0)
            data = test_data
        
        data_files = {
            ReadDataABC.BUREAU_NAME: self.get_aggregated_bureau_data,
            ReadDataABC.CREDIT_CARD_BALANCE_NAME: self.get_aggregated_credit_card_balance_data,
            ReadDataABC.INSTALLMENTS_PAYMENTS_NAME: self.get_aggregated_installments_payments_data,
            ReadDataABC.PREVIOUS_APPLICATION_NAME: self.get_aggregated_previous_application_data,
            ReadDataABC.POS_CASH_BALANCE_NAME: self.get_aggregated_pos_cash_balance_data
        }

        for file_name, aggregation_method in data_files.items():
            temp_data = pd.read_csv(f"{files_path}/{file_name}")
            temp_data = temp_data[temp_data['SK_ID_CURR'].isin(data['SK_ID_CURR'])]
            aggregated_data = aggregation_method(temp_data)
            data = pd.merge(data, aggregated_data, on="SK_ID_CURR", how="outer")
     
        # Concatenate all the data into a single DataFrame
        return data
    
    def write_data(self, files_path : str, filename: str, sampling_frequency: int = 1, training : bool = True):
        """
        Write the data for the model.
        It is a merge of the training data and the aggregated data from the other tables.

        Parameters:
        files_path (str): The path where the file are located.
        filename (str): The name of the file to write.
        """
        
        data = self.retrieve_data(files_path, sampling_frequency=sampling_frequency, training=training)
        data.drop(columns=['SK_ID_CURR'], inplace=True)
    
        data.to_csv(f"{files_path}/{filename}", index=False)

    def read_data(self, file_path: str, filename: str):
        """
        Read data from a CSV file.

        This method reads the file specified by filename from the directory specified by file_path.
        It assumes that the file is in CSV format.

        Parameters:
        file_path (str): The path where the file is located.
        filename (str): The name of the file to read.

        Returns:
        pd.DataFrame: The data read from the file as a pandas DataFrame.
        """
        # Read the data from the file
        data = pd.read_csv(f"{file_path}/{filename}")

        # Return the data
        return data
    
    def write_data_structure_json(self, data: pd.DataFrame, file_path: str, filename: str):
        """
        Write the data structure to a JSON file.

        This method writes the data structure to a JSON file.
        It assumes that the file is in JSON format.

        Parameters:
        file_path (str): The path where the file is located.
        filename (str): The name of the file to write.
        """
    
        schema = {}
        for col in data.columns:
            # Determine the datatype of the column
            dtype = str(data[col].dtype)

            # If the datatype is numerical or there are too many unique values, skip the values list
            if (dtype.startswith('int') and data[col].nunique() > 2 ) or dtype.startswith('float') :
                min = int(data[col].min())
                max = int(data[col].max())

                max = max if max-min > 1 else max+1

                schema[col] = {
                    'type': dtype,
                    'min': min,
                    'max': max,
                }
            else:
                # For non-numerical types
                unique_values = data[col].dropna().unique()
               
                schema[col] = {
                    'type': dtype,
                    'values': unique_values.tolist()
                }

        with open(f"{file_path}/{filename}", 'w') as f:
            json.dump(schema, f, indent=4)

