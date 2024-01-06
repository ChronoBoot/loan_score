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

        # Calculte the duration of the credit
        # We either use the factual end date or the expected end date (if credit is not closed)
        bureau_data = bureau_data.assign(
            CREDIT_DURATION = bureau_data['DAYS_CREDIT'] - np.where(
                bureau_data['DAYS_ENDDATE_FACT'].isnull(),
                bureau_data['DAYS_CREDIT_ENDDATE'],
                bureau_data['DAYS_ENDDATE_FACT']
            )
        )

        # One-hot encoding on CREDIT_ACTIVE
        bureau_data_active = pd.get_dummies(bureau_data['CREDIT_ACTIVE'], prefix='CREDIT_ACTIVE')
        bureau_data = pd.concat([bureau_data, bureau_data_active], axis=1)

        # One-hot encoding on CREDIT_CURRENCY
        bureau_data_currency = pd.get_dummies(bureau_data['CREDIT_CURRENCY'], prefix='CREDIT_CURRENCY')
        bureau_data = pd.concat([bureau_data, bureau_data_currency], axis=1)

        # One-hot encoding on CREDIT_TYPE
        bureau_data_credit_types = pd.get_dummies(bureau_data['CREDIT_TYPE'], prefix='CREDIT_TYPE')       
        bureau_data = pd.concat([bureau_data, bureau_data_credit_types], axis=1)

        aggreagation_dict = {
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

        credit_active_count_dict = {
            col: pd.Series.count for col in bureau_data.columns if col.startswith('CREDIT_ACTIVE_')
        }

        aggreagation_dict.update(credit_active_count_dict)

        credit_currency_count_dict = {
            col: pd.Series.count for col in bureau_data.columns if col.startswith('CREDIT_CURRENCY_')
        }

        aggreagation_dict.update(credit_currency_count_dict)

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
    
    def get_aggregated_credit_card_balance_data(self, credit_card_balance_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate the data from credit_card_balance.

        Parameters:
        credit_card_balance_data (pd.DataFrame): The data from the credit_card_balance table.

        Returns:
        pd.DataFrame: The aggregated data. Grouped by SK_ID_CURR.
        """

        # One-hot encoding on NAME_CONTRACT_STATUS
        credit_card_balance_data_status = pd.get_dummies(credit_card_balance_data['NAME_CONTRACT_STATUS'], prefix='NAME_CONTRACT_STATUS')
        credit_card_balance_data = pd.concat([credit_card_balance_data, credit_card_balance_data_status], axis=1)

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

        credit_card_balance_data_aggregation_dict = {
            col: pd.Series.count for col in credit_card_balance_data.columns if col.startswith('NAME_CONTRACT_STATUS_')
        }

        aggregate_dict.update(credit_card_balance_data_aggregation_dict)

        aggregated_credit_card_balance_data = credit_card_balance_data.groupby("SK_ID_CURR").agg(aggregate_dict)

        # Flatten the MultiIndex columns
        aggregated_credit_card_balance_data.columns = ['_'.join(col).strip() for col in aggregated_credit_card_balance_data.columns.values]

        # Reset the index
        aggregated_credit_card_balance_data.reset_index(inplace=True)

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

        aggregated_installments_payments_data = installments_payments_data.groupby("SK_ID_CURR").agg(aggregate_dict)

        # Flatten the MultiIndex columns
        aggregated_installments_payments_data.columns = ['_'.join(col).strip() for col in aggregated_installments_payments_data.columns.values]

        # Reset the index
        aggregated_installments_payments_data.reset_index(inplace=True)

        return aggregated_installments_payments_data
    
    def get_aggregated_previous_application_data(self, previous_application_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate the data from previous_application.

        Parameters:
        previous_application_data (pd.DataFrame): The data from the previous_application table.

        Returns:
        pd.DataFrame: The aggregated data. Grouped by SK_ID_CURR.
        """
        
        # One-hot encoding on NAME_CONTRACT_TYPE
        previous_application_data_contract_type = pd.get_dummies(previous_application_data['NAME_CONTRACT_TYPE'], prefix='NAME_CONTRACT_TYPE')
        previous_application_data = pd.concat([previous_application_data, previous_application_data_contract_type], axis=1)

        # One-hot encoding on WEEKDAY_APPR_PROCESS_START
        previous_application_data_weekday = pd.get_dummies(previous_application_data['WEEKDAY_APPR_PROCESS_START'], prefix='WEEKDAY_APPR_PROCESS_START')
        previous_application_data = pd.concat([previous_application_data, previous_application_data_weekday], axis=1)

        # One-hot encoding on FLAG_LAST_APPL_PER_CONTRACT
        previous_application_data_last_appl = pd.get_dummies(previous_application_data['FLAG_LAST_APPL_PER_CONTRACT'], prefix='FLAG_LAST_APPL_PER_CONTRACT')
        previous_application_data = pd.concat([previous_application_data, previous_application_data_last_appl], axis=1)

        # One-hot encoding on NFLAG_LAST_APPL_IN_DAY
        previous_application_data_last_appl_day = pd.get_dummies(previous_application_data['NFLAG_LAST_APPL_IN_DAY'], prefix='NFLAG_LAST_APPL_IN_DAY')
        previous_application_data = pd.concat([previous_application_data, previous_application_data_last_appl_day], axis=1)

        # One-hot encoding on NAME_CASH_LOAN_PURPOSE
        previous_application_data_purpose = pd.get_dummies(previous_application_data['NAME_CASH_LOAN_PURPOSE'], prefix='NAME_CASH_LOAN_PURPOSE')
        previous_application_data = pd.concat([previous_application_data, previous_application_data_purpose], axis=1)

        # One-hot encoding on NAME_CONTRACT_STATUS
        previous_application_data_status = pd.get_dummies(previous_application_data['NAME_CONTRACT_STATUS'], prefix='NAME_CONTRACT_STATUS')
        previous_application_data = pd.concat([previous_application_data, previous_application_data_status], axis=1)

        # One-hot encoding on NAME_PAYMENT_TYPE
        previous_application_data_payment_type = pd.get_dummies(previous_application_data['NAME_PAYMENT_TYPE'], prefix='NAME_PAYMENT_TYPE')
        previous_application_data = pd.concat([previous_application_data, previous_application_data_payment_type], axis=1)

        # One-hot encoding on CODE_REJECT_REASON
        previous_application_data_reject_reason = pd.get_dummies(previous_application_data['CODE_REJECT_REASON'], prefix='CODE_REJECT_REASON')
        previous_application_data = pd.concat([previous_application_data, previous_application_data_reject_reason], axis=1)

        # One-hot encoding on NAME_TYPE_SUITE
        previous_application_data_suite = pd.get_dummies(previous_application_data['NAME_TYPE_SUITE'], prefix='NAME_TYPE_SUITE')
        previous_application_data = pd.concat([previous_application_data, previous_application_data_suite], axis=1)

        # One-hot encoding on NAME_CLIENT_TYPE
        previous_application_data_client_type = pd.get_dummies(previous_application_data['NAME_CLIENT_TYPE'], prefix='NAME_CLIENT_TYPE')
        previous_application_data = pd.concat([previous_application_data, previous_application_data_client_type], axis=1)

        # One-hot encoding on NAME_GOODS_CATEGORY
        previous_application_data_goods_category = pd.get_dummies(previous_application_data['NAME_GOODS_CATEGORY'], prefix='NAME_GOODS_CATEGORY')
        previous_application_data = pd.concat([previous_application_data, previous_application_data_goods_category], axis=1)

        # One-hot encoding on NAME_PORTFOLIO
        previous_application_data_portfolio = pd.get_dummies(previous_application_data['NAME_PORTFOLIO'], prefix='NAME_PORTFOLIO')
        previous_application_data = pd.concat([previous_application_data, previous_application_data_portfolio], axis=1)

        # One-hot encoding on NAME_PRODUCT_TYPE
        previous_application_data_product_type = pd.get_dummies(previous_application_data['NAME_PRODUCT_TYPE'], prefix='NAME_PRODUCT_TYPE')
        previous_application_data = pd.concat([previous_application_data, previous_application_data_product_type], axis=1)

        # One-hot encoding on CHANNEL_TYPE
        previous_application_data_channel_type = pd.get_dummies(previous_application_data['CHANNEL_TYPE'], prefix='CHANNEL_TYPE')
        previous_application_data = pd.concat([previous_application_data, previous_application_data_channel_type], axis=1)

        # One-hot encoding on NAME_SELLER_INDUSTRY
        previous_application_data_seller_industry = pd.get_dummies(previous_application_data['NAME_SELLER_INDUSTRY'], prefix='NAME_SELLER_INDUSTRY')
        previous_application_data = pd.concat([previous_application_data, previous_application_data_seller_industry], axis=1)

        # One-hot encoding on NAME_YIELD_GROUP
        previous_application_data_yield_group = pd.get_dummies(previous_application_data['NAME_YIELD_GROUP'], prefix='NAME_YIELD_GROUP')
        previous_application_data = pd.concat([previous_application_data, previous_application_data_yield_group], axis=1)

        # One-hot encoding on PRODUCT_COMBINATION
        previous_application_data_product_combination = pd.get_dummies(previous_application_data['PRODUCT_COMBINATION'], prefix='PRODUCT_COMBINATION')
        previous_application_data = pd.concat([previous_application_data, previous_application_data_product_combination], axis=1)

        # One-hot encoding on NFLAG_INSURED_ON_APPROVAL
        previous_application_data_insured = pd.get_dummies(previous_application_data['NFLAG_INSURED_ON_APPROVAL'], prefix='NFLAG_INSURED_ON_APPROVAL')
        previous_application_data = pd.concat([previous_application_data, previous_application_data_insured], axis=1)

        aggregate_dict = {
            'AMT_ANNUITY': ['max', 'min', 'mean'],
            'AMT_APPLICATION': ['max', 'min', 'mean'],
            'AMT_CREDIT': ['max', 'min', 'mean'],
            'AMT_DOWN_PAYMENT': ['max', 'min', 'mean'],
            'AMT_GOODS_PRICE': ['max', 'min', 'mean'],
            'HOUR_APPR_PROCESS_START': ['max', 'min', 'mean'],
            'RATE_DOWN_PAYMENT': ['max', 'min', 'mean'],
            'DAYS_DECISION': ['max', 'min'],
            'SELLERPLACE_AREA': ['max', 'min', 'mean'],
            'CNT_PAYMENT': ['max', 'min', 'mean'],
            'DAYS_FIRST_DRAWING': ['max', 'min'],
            'DAYS_FIRST_DUE': ['max', 'min'],
            'DAYS_LAST_DUE_1ST_VERSION': ['max', 'min'],
            'DAYS_LAST_DUE': ['max', 'min'],
            'DAYS_TERMINATION': ['max', 'min'],
        }

        contract_type_count_dict = {
            col: pd.Series.count for col in previous_application_data.columns if col.startswith('NAME_CONTRACT_TYPE_')
        }

        aggregate_dict.update(contract_type_count_dict)

        weekday_count_dict = {
            col: pd.Series.count for col in previous_application_data.columns if col.startswith('WEEKDAY_APPR_PROCESS_START_')
        }

        aggregate_dict.update(weekday_count_dict)

        last_appl_count_dict = {
            col: pd.Series.count for col in previous_application_data.columns if col.startswith('FLAG_LAST_APPL_PER_CONTRACT_')
        }

        aggregate_dict.update(last_appl_count_dict)

        last_appl_day_count_dict = {
            col: pd.Series.count for col in previous_application_data.columns if col.startswith('NFLAG_LAST_APPL_IN_DAY_')
        }

        aggregate_dict.update(last_appl_day_count_dict)

        purpose_count_dict = {
            col: pd.Series.count for col in previous_application_data.columns if col.startswith('NAME_CASH_LOAN_PURPOSE_')
        }

        aggregate_dict.update(purpose_count_dict)

        status_count_dict = {
            col: pd.Series.count for col in previous_application_data.columns if col.startswith('NAME_CONTRACT_STATUS_')
        }

        aggregate_dict.update(status_count_dict)

        payment_type_count_dict = {
            col: pd.Series.count for col in previous_application_data.columns if col.startswith('NAME_PAYMENT_TYPE_')
        }

        aggregate_dict.update(payment_type_count_dict)

        reject_reason_count_dict = {
            col: pd.Series.count for col in previous_application_data.columns if col.startswith('CODE_REJECT_REASON_')
        }

        aggregate_dict.update(reject_reason_count_dict)

        suite_count_dict = {
            col: pd.Series.count for col in previous_application_data.columns if col.startswith('NAME_TYPE_SUITE_')
        }

        aggregate_dict.update(suite_count_dict)

        client_type_count_dict = {
            col: pd.Series.count for col in previous_application_data.columns if col.startswith('NAME_CLIENT_TYPE_')
        }

        aggregate_dict.update(client_type_count_dict)

        goods_category_count_dict = {
            col: pd.Series.count for col in previous_application_data.columns if col.startswith('NAME_GOODS_CATEGORY_')
        }

        aggregate_dict.update(goods_category_count_dict)

        portfolio_count_dict = {
            col: pd.Series.count for col in previous_application_data.columns if col.startswith('NAME_PORTFOLIO_')
        }

        aggregate_dict.update(portfolio_count_dict)

        product_type_count_dict = {
            col: pd.Series.count for col in previous_application_data.columns if col.startswith('NAME_PRODUCT_TYPE_')
        }

        aggregate_dict.update(product_type_count_dict)

        channel_type_count_dict = {
            col: pd.Series.count for col in previous_application_data.columns if col.startswith('CHANNEL_TYPE_')
        }

        aggregate_dict.update(channel_type_count_dict)

        seller_industry_count_dict = {
            col: pd.Series.count for col in previous_application_data.columns if col.startswith('NAME_SELLER_INDUSTRY_')
        }

        aggregate_dict.update(seller_industry_count_dict)

        yield_group_count_dict = {
            col: pd.Series.count for col in previous_application_data.columns if col.startswith('NAME_YIELD_GROUP_')
        }

        aggregate_dict.update(yield_group_count_dict)

        product_combination_count_dict = {
            col: pd.Series.count for col in previous_application_data.columns if col.startswith('PRODUCT_COMBINATION_')
        }

        aggregate_dict.update(product_combination_count_dict)

        insured_count_dict = {
            col: pd.Series.count for col in previous_application_data.columns if col.startswith('NFLAG_INSURED_ON_APPROVAL_')
        }

        aggregate_dict.update(insured_count_dict)

        aggregated_previous_application_data = previous_application_data.groupby("SK_ID_CURR").agg(aggregate_dict)

        # Flatten the MultiIndex columns
        aggregated_previous_application_data.columns = ['_'.join(col).strip() for col in aggregated_previous_application_data.columns.values]

        # Reset the index
        aggregated_previous_application_data.reset_index(inplace=True)

        return aggregated_previous_application_data

    def get_aggregated_pos_cash_balance_data(self, pos_cash_balance_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate the data from pos_cash_balance.

        Parameters:
        pos_cash_balance_data (pd.DataFrame): The data from the pos_cash_balance table.

        Returns:
        pd.DataFrame: The aggregated data. Grouped by SK_ID_CURR.
        """
        
        pos_cash_balance_data_contract_status = pd.get_dummies(pos_cash_balance_data['NAME_CONTRACT_STATUS'], prefix='NAME_CONTRACT_STATUS')
        pos_cash_balance_data = pd.concat([pos_cash_balance_data, pos_cash_balance_data_contract_status], axis=1)

        aggregate_dict = {
            'MONTHS_BALANCE': ['max', 'min'],
            'CNT_INSTALMENT': ['max', 'min', 'mean'],
            'CNT_INSTALMENT_FUTURE': ['max', 'min', 'mean'],
            'SK_DPD': ['max', 'min', 'mean'],
            'SK_DPD_DEF': ['max', 'min', 'mean']
        }

        contract_status_count_dict = {
            col: pd.Series.count for col in pos_cash_balance_data.columns if col.startswith('NAME_CONTRACT_STATUS_')
        }

        aggregate_dict.update(contract_status_count_dict)

        aggregated_pos_cash_balance_data = pos_cash_balance_data.groupby("SK_ID_CURR").agg(aggregate_dict)

        # Flatten the MultiIndex columns
        aggregated_pos_cash_balance_data.columns = ['_'.join(col).strip() for col in aggregated_pos_cash_balance_data.columns.values]

        # Reset the index
        aggregated_pos_cash_balance_data.reset_index(inplace=True)

        return aggregated_pos_cash_balance_data
    
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

            credit_card_balance_data = pd.read_csv(f"{files_path}/{ReadDataABC.CREDIT_CARD_BALANCE_NAME}")
            credit_card_balance_data = credit_card_balance_data[credit_card_balance_data['SK_ID_CURR'].isin(train_data['SK_ID_CURR'])]

            installments_payments_data = pd.read_csv(f"{files_path}/{ReadDataABC.INSTALLMENTS_PAYMENTS_NAME}")
            installments_payments_data = installments_payments_data[installments_payments_data['SK_ID_CURR'].isin(train_data['SK_ID_CURR'])]

            previous_application_data = pd.read_csv(f"{files_path}/{ReadDataABC.PREVIOUS_APPLICATION_NAME}")
            previous_application_data = previous_application_data[previous_application_data['SK_ID_CURR'].isin(train_data['SK_ID_CURR'])]

            pos_cash_balance_data = pd.read_csv(f"{files_path}/{ReadDataABC.POS_CASH_BALANCE_NAME}")
            pos_cash_balance_data = pos_cash_balance_data[pos_cash_balance_data['SK_ID_CURR'].isin(train_data['SK_ID_CURR'])]


            # Merge the main table with the rest of the tables
            # Merge application_train with aggragated bureau
            aggregated_bureau_data = self.get_aggregated_bureau_data(bureau_data)
            data = pd.merge(train_data, aggregated_bureau_data, on="SK_ID_CURR")

            # Merge application_train with credit_card_balance
            aggregated_credit_card_balance_data = self.get_aggregated_credit_card_balance_data(credit_card_balance_data)
            data = pd.merge(data, aggregated_credit_card_balance_data, on="SK_ID_CURR")

            # Merge application_train with installments_payments
            aggregated_installments_payments_data = self.get_aggregated_installments_payments_data(installments_payments_data)
            data = pd.merge(data, aggregated_installments_payments_data, on="SK_ID_CURR")

            # Merge application_train with previous_application
            aggregated_previous_application_data = self.get_aggregated_previous_application_data(previous_application_data)
            data = pd.merge(data, aggregated_previous_application_data, on="SK_ID_CURR")

            # Merge application_train with pos_cash_balance
            aggregated_pos_cash_balance_data = self.get_aggregated_pos_cash_balance_data(pos_cash_balance_data)
            data = pd.merge(data, aggregated_pos_cash_balance_data, on="SK_ID_CURR")
        else:
            data = train_data
            
        # Concatenate all the data into a single DataFrame
        return data