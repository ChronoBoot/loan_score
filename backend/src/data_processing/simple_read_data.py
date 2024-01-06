import numpy as np
import pandas as pd
from backend.src.data_processing.read_data_abc import ReadDataABC

class SimpleReadData(ReadDataABC):
    """
    Simple implementation of the ReadDataABC abstract base class.

    This class provides a simple way to read data from CSV files.
    """

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

        # Split the credit active column into 2 columns : IS_ACTIVE and IS_CLOSED
        bureau_data = bureau_data.assign(
            IS_ACTIVE = bureau_data['CREDIT_ACTIVE'] == 'Active',
            IS_CLOSED = bureau_data['CREDIT_ACTIVE'] == 'Closed',
        )

        # Calculte the duration of the credit
        # We either use the factual end date or the expected end date (if credit is not closed)
        bureau_data = bureau_data.assign(
            CREDIT_DURATION = bureau_data['DAYS_CREDIT'] - np.where(
                bureau_data['DAYS_ENDDATE_FACT'].isnull(),
                bureau_data['DAYS_CREDIT_ENDDATE'],
                bureau_data['DAYS_ENDDATE_FACT']
            )
        )

        # Fill missing values for AMT_CREDIT_MAX_OVERDUE
        bureau_data['AMT_CREDIT_MAX_OVERDUE'].fillna(0, inplace=True) 

        # One-hot encoding on CREDIT_TYPE
        bureau_data_credit_types = pd.get_dummies(bureau_data['CREDIT_TYPE'], prefix='CREDIT_TYPE_')       
        bureau_data = pd.concat([bureau_data, bureau_data_credit_types], axis=1)

        aggreagation_dict = {
            'IS_ACTIVE': ['count'],
            'IS_CLOSED': ['count'],
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

        credit_type_count_dict = {
            col: pd.Series.count for col in bureau_data.columns if col.startswith('CREDIT_TYPE_')
        }

        aggreagation_dict.update(credit_type_count_dict)

        aggregated_bureau_data = bureau_data.groupby("SK_ID_CURR").agg(aggreagation_dict)

        # Flatten the MultiIndex columns
        aggregated_bureau_data.columns = ['_'.join(col).strip() for col in aggregated_bureau_data.columns.values]

        # Reset the index
        aggregated_bureau_data.reset_index(inplace=True) 

        aggregated_bureau_data = aggregated_bureau_data.assign(DAYS_CREDIT_DIFF_MEAN = bureau_data_days_credit_diff_mean)

        return aggregated_bureau_data

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

        train_data = pd.read_csv(f"{files_path}/{ReadDataABC.APPLICATION_TRAIN_NAME}", skiprows=lambda i: i % sampling_frequency != 0)

        if concat:
            # Read the rest of the files and filtered to only keep the rows that are in the main file
            bureau_data = pd.read_csv(f"{files_path}/{ReadDataABC.BUREAU_NAME}")
            bureau_data = bureau_data[bureau_data['SK_ID_CURR'].isin(train_data['SK_ID_CURR'])]

            bureau_balance_data = pd.read_csv(f"{files_path}/{ReadDataABC.BUREAU_BALANCE_NAME}")
            bureau_balance_data = bureau_balance_data[bureau_balance_data['SK_ID_BUREAU'].isin(bureau_data['SK_ID_BUREAU'])]

            credit_card_balance_data = pd.read_csv(f"{files_path}/{ReadDataABC.CREDIT_CARD_BALANCE_NAME}")
            credit_card_balance_data = credit_card_balance_data[credit_card_balance_data['SK_ID_CURR'].isin(train_data['SK_ID_CURR'])]

            installments_payments_data = pd.read_csv(f"{files_path}/{ReadDataABC.INSTALLMENTS_PAYMENTS_NAME}")
            installments_payments_data = installments_payments_data[installments_payments_data['SK_ID_CURR'].isin(train_data['SK_ID_CURR'])]

            previous_application_data = pd.read_csv(f"{files_path}/{ReadDataABC.PREVIOUS_APPLICATION_NAME}")
            previous_application_data = previous_application_data[previous_application_data['SK_ID_CURR'].isin(train_data['SK_ID_CURR'])]

            pos_cash_balance_data = pd.read_csv(f"{files_path}/{ReadDataABC.POS_CASH_BALANCE_NAME}")
            pos_cash_balance_data = pos_cash_balance_data[pos_cash_balance_data['SK_ID_CURR'].isin(train_data['SK_ID_CURR'])]


            aggregated_bureau_data = self.get_aggregated_bureau_data(bureau_data)

            # Merge the main table with the rest of the tables
            # Merge application_train with aggragated bureau
            data = pd.merge(train_data, aggregated_bureau_data, on="SK_ID_CURR")

            # Merge bureau with bureau_balance
            # data = pd.merge(data, bureau_balance_data, on="SK_ID_BUREAU")

            # # Merge application_train with credit_card_balance
            # data = pd.merge(data, credit_card_balance_data, on="SK_ID_CURR")

            # # Merge application_train with installments_payments
            # data = pd.merge(data, installments_payments_data, on="SK_ID_CURR")

            # # Merge application_train with previous_application
            # data = pd.merge(data, previous_application_data, on="SK_ID_CURR")

            # # Merge application_train with pos_cash_balance
            # data = pd.merge(data, pos_cash_balance_data, on="SK_ID_CURR")
        else:
            data = train_data
            
        # Concatenate all the data into a single DataFrame
        return data