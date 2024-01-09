import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from backend.src.data_processing.simple_read_data import SimpleReadData

class TestSimpleReadData(unittest.TestCase):
    def setUp(self):
        self.reader = SimpleReadData()

    def test_one_hot_encode(self):
        # Arrange
        data = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']})
        columns_names = ['A', 'B']

        # Act
        result = self.reader.one_hot_encode(data, columns_names)

        # Assert
        expected_result = pd.DataFrame({
            'A': ['a', 'b', 'c'], 
            'B': ['x', 'y', 'z'], 
            'A_a': [True, False, False], 
            'A_b': [False, True, False], 
            'A_c': [False, False, True], 
            'B_x': [True, False, False], 
            'B_y': [False, True, False], 
            'B_z': [False, False, True]
            })
        pd.testing.assert_frame_equal(result, expected_result)

    def test_update_aggregation_dict(self):
        # Arrange
        data = pd.DataFrame({
            'A': [1, 2, 3, 3, 2, 1], 
            'A_1' : [True, False, False, False, False, True], 
            'A_2' : [False, True, False, False, True, False], 
            'A_3' : [False, False, True, True, False, False],
        })
        aggregation_dict = {}
        prefixes = ['A']

        # Act
        self.reader.update_aggregation_dict(data, aggregation_dict, prefixes)

        # Assert
        expected_result = {
            'A_1': ['sum'],
            'A_2': ['sum'],
            'A_3': ['sum'],
        }
        self.assertEqual(aggregation_dict, expected_result)

    def test_flatten_and_reset_index(self):
        # Arrange
        data = {
            'ID': [1, 2, 3],
            ('Category A', 'Subcategory 1'): [1, 2, 3],
            ('Category A', 'Subcategory 2'): [4, 5, 6],
            ('Category B', 'Subcategory 1'): [7, 8, 9],
            ('Category B', 'Subcategory 2'): [10, 11, 12]
        }
        multi_level_col_df = pd.DataFrame(data)
        multi_level_col_df.set_index('ID', inplace=True)

        # Act
        self.reader.flatten_and_reset_index(multi_level_col_df)

        # Assert
        expected_result = pd.DataFrame({
            'ID': [1, 2, 3],
            'Category A_Subcategory 1': [1, 2, 3],
            'Category A_Subcategory 2': [4, 5, 6],
            'Category B_Subcategory 1': [7, 8, 9],
            'Category B_Subcategory 2': [10, 11, 12]
        })
        pd.testing.assert_frame_equal(multi_level_col_df, expected_result)

    def test_aggregate_data(self):
        # Arrange
        data = pd.DataFrame({
            'ID': [1, 2, 3, 1, 2, 3],
            'A': [1, 2, 3, 2, 7, 8], 
            'B': [4, 5, 6, 0, 2, 4],
            'C': ['a', 'b', 'c', 'c', 'b', 'a'],
            'C_a': [True, False, False, False, False, True],
            'C_b': [False, True, False, False, True, False],
            'C_c': [False, False, True, True, False, False], 
        })
        aggregation_dict = {'A': ['max', 'min'], 'B': ['mean']}
        prefixes = ['C']
        groupby_col = 'ID'

        # Act
        result = self.reader.aggregate_data(data, aggregation_dict, prefixes, groupby_col)

        # Assert
        expected_result = pd.DataFrame({
            'ID': [1, 2, 3], 
            'A_max': [2, 7, 8], 
            'A_min': [1, 2, 3], 
            'B_mean': [2.0, 3.5, 5.0],
            'C_a_sum': [1, 0, 1],
            'C_b_sum': [0, 2, 0],
            'C_c_sum': [1, 0, 1],
        })
        pd.testing.assert_frame_equal(result, expected_result)

    def test_get_aggregated_bureau_data(self):
        # Arrange
        bureau_data = pd.DataFrame({
            "SK_ID_CURR": [123456, 123456],
            "SK_ID_BUREAU": [654321, 654322],
            "CREDIT_ACTIVE": ["Active", "Closed"],
            "CREDIT_CURRENCY": ["currency1", "currency2"],
            "DAYS_CREDIT": [-500, -150],
            "CREDIT_DAY_OVERDUE": [0, 0],
            "DAYS_CREDIT_ENDDATE": [-365, -50],
            "DAYS_ENDDATE_FACT": [None, -100],
            "AMT_CREDIT_MAX_OVERDUE": [1000.0, None],
            "CNT_CREDIT_PROLONG": [0, 0],
            "AMT_CREDIT_SUM": [50000.0, 100000.0],
            "AMT_CREDIT_SUM_DEBT": [25000.0, 0.0],
            "AMT_CREDIT_SUM_LIMIT": [0.0, 5000.0],
            "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0],
            "CREDIT_TYPE": ["Consumer credit", "Credit card"],
            "DAYS_CREDIT_UPDATE": [-30, -15],
            "AMT_ANNUITY": [2500.0, 5000.0]
        })

        # Act
        result = self.reader.get_aggregated_bureau_data(bureau_data)

        # Assert
        expected_result = pd.DataFrame({
            'SK_ID_CURR': [123456],
            'DAYS_CREDIT_max': [-150],
            'DAYS_CREDIT_min': [-500],
            'CREDIT_DAY_OVERDUE_max': [0],
            'CREDIT_DAY_OVERDUE_min': [0],
            'CREDIT_DAY_OVERDUE_mean': [0.0],
            'CREDIT_DURATION_max': [135],
            'CREDIT_DURATION_min': [50],
            'CREDIT_DURATION_mean': [92.5],
            'AMT_CREDIT_MAX_OVERDUE_max': [1000.0],
            'AMT_CREDIT_MAX_OVERDUE_min': [1000.0],
            'AMT_CREDIT_MAX_OVERDUE_mean': [1000.0],
            'CNT_CREDIT_PROLONG_max': [0],
            'CNT_CREDIT_PROLONG_min': [0],
            'CNT_CREDIT_PROLONG_mean': [0.0],
            'AMT_CREDIT_SUM_max': [100000.0],
            'AMT_CREDIT_SUM_min': [50000.0],
            'AMT_CREDIT_SUM_mean': [75000.0],
            'AMT_CREDIT_SUM_DEBT_max': [25000.0],
            'AMT_CREDIT_SUM_DEBT_min': [0.0],
            'AMT_CREDIT_SUM_DEBT_mean': [12500.0],
            'AMT_CREDIT_SUM_LIMIT_max': [5000.0],
            'AMT_CREDIT_SUM_LIMIT_min': [0.0],
            'AMT_CREDIT_SUM_LIMIT_mean': [2500.0],
            'AMT_CREDIT_SUM_OVERDUE_max': [0.0],
            'AMT_CREDIT_SUM_OVERDUE_min': [0.0],
            'AMT_CREDIT_SUM_OVERDUE_mean': [0.0],
            'DAYS_CREDIT_UPDATE_min': [-30],
            'AMT_ANNUITY_max': [5000.0],
            'AMT_ANNUITY_min': [2500.0],
            'AMT_ANNUITY_mean': [3750.0],
            'CREDIT_ACTIVE_Active_sum': [1],
            'CREDIT_ACTIVE_Closed_sum': [1],
            'CREDIT_CURRENCY_currency1_sum': [1],
            'CREDIT_CURRENCY_currency2_sum': [1],
            'CREDIT_TYPE_Consumer credit_sum': [1],
            'CREDIT_TYPE_Credit card_sum': [1],
            'DAYS_CREDIT_DIFF_MEAN': [350.0],
        })
        pd.testing.assert_frame_equal(result, expected_result)

    def test_get_aggregated_credit_card_balance_data(self):
        # Arrange
        credit_card_balance_data = pd.DataFrame({
            "SK_ID_PREV": [378903, 378904],
            "SK_ID_CURR": [1004195, 1004195],
            "MONTHS_BALANCE": [-2, -11],
            "AMT_BALANCE": [25000, 30000],
            "AMT_CREDIT_LIMIT_ACTUAL": [45000, 50000],
            "AMT_DRAWINGS_ATM_CURRENT": [5000, 4000],
            "AMT_DRAWINGS_CURRENT": [7000, 6000],
            "AMT_DRAWINGS_OTHER_CURRENT": [2000, 1500],
            "AMT_DRAWINGS_POS_CURRENT": [5000, 4500],
            "AMT_INST_MIN_REGULARITY": [3500, 4000],
            "AMT_PAYMENT_CURRENT": [5000, 4500],
            "AMT_PAYMENT_TOTAL_CURRENT": [5500, 5000],
            "AMT_RECEIVABLE_PRINCIPAL": [24000, 29000],
            "AMT_RECIVABLE": [25000, 30000],
            "AMT_TOTAL_RECEIVABLE": [25000, 30000],
            "CNT_DRAWINGS_ATM_CURRENT": [2, 1],
            "CNT_DRAWINGS_CURRENT": [3, 2],
            "CNT_DRAWINGS_OTHER_CURRENT": [1, 0],
            "CNT_DRAWINGS_POS_CURRENT": [2, 1],
            "CNT_INSTALMENT_MATURE_CUM": [26, 26],
            "NAME_CONTRACT_STATUS": ["Active", "Active"],
            "SK_DPD": [5, 2],
            "SK_DPD_DEF": [3, 1]
        })

        # Act
        result = self.reader.get_aggregated_credit_card_balance_data(credit_card_balance_data)

        # Assert
        expected_result = pd.DataFrame({
            'SK_ID_CURR': [1004195],
            'MONTHS_BALANCE_max': [-2],
            'MONTHS_BALANCE_min': [-11],
            'AMT_BALANCE_max': [30000],
            'AMT_BALANCE_min': [25000],
            'AMT_BALANCE_mean': [27500.0],
            'AMT_CREDIT_LIMIT_ACTUAL_max': [50000],
            'AMT_CREDIT_LIMIT_ACTUAL_min': [45000],
            'AMT_CREDIT_LIMIT_ACTUAL_mean': [47500.0],
            'AMT_DRAWINGS_ATM_CURRENT_max': [5000],
            'AMT_DRAWINGS_ATM_CURRENT_min': [4000],
            'AMT_DRAWINGS_ATM_CURRENT_mean': [4500.0],
            'AMT_DRAWINGS_CURRENT_max': [7000],
            'AMT_DRAWINGS_CURRENT_min': [6000],
            'AMT_DRAWINGS_CURRENT_mean': [6500.0],
            'AMT_DRAWINGS_OTHER_CURRENT_max': [2000],
            'AMT_DRAWINGS_OTHER_CURRENT_min': [1500],
            'AMT_DRAWINGS_OTHER_CURRENT_mean': [1750.0],
            'AMT_DRAWINGS_POS_CURRENT_max': [5000],
            'AMT_DRAWINGS_POS_CURRENT_min': [4500],
            'AMT_DRAWINGS_POS_CURRENT_mean': [4750.0],
            'AMT_INST_MIN_REGULARITY_max': [4000],
            'AMT_INST_MIN_REGULARITY_min': [3500],
            'AMT_INST_MIN_REGULARITY_mean': [3750.0],
            'AMT_PAYMENT_CURRENT_max': [5000],
            'AMT_PAYMENT_CURRENT_min': [4500],
            'AMT_PAYMENT_CURRENT_mean': [4750.0],
            'AMT_PAYMENT_TOTAL_CURRENT_max': [5500],
            'AMT_PAYMENT_TOTAL_CURRENT_min': [5000],
            'AMT_PAYMENT_TOTAL_CURRENT_mean': [5250.0],
            'AMT_RECEIVABLE_PRINCIPAL_max': [29000],
            'AMT_RECEIVABLE_PRINCIPAL_min': [24000],
            'AMT_RECEIVABLE_PRINCIPAL_mean': [26500.0],
            'AMT_RECIVABLE_max': [30000],
            'AMT_RECIVABLE_min': [25000],
            'AMT_RECIVABLE_mean': [27500.0],
            'AMT_TOTAL_RECEIVABLE_max': [30000],
            'AMT_TOTAL_RECEIVABLE_min': [25000],
            'AMT_TOTAL_RECEIVABLE_mean': [27500.0],
            'CNT_DRAWINGS_ATM_CURRENT_max': [2],
            'CNT_DRAWINGS_ATM_CURRENT_min': [1],
            'CNT_DRAWINGS_ATM_CURRENT_mean': [1.5],
            'CNT_DRAWINGS_CURRENT_max': [3],
            'CNT_DRAWINGS_CURRENT_min': [2],
            'CNT_DRAWINGS_CURRENT_mean': [2.5],
            'CNT_DRAWINGS_OTHER_CURRENT_max': [1],
            'CNT_DRAWINGS_OTHER_CURRENT_min': [0],
            'CNT_DRAWINGS_OTHER_CURRENT_mean': [0.5],
            'CNT_DRAWINGS_POS_CURRENT_max': [2],
            'CNT_DRAWINGS_POS_CURRENT_min': [1],
            'CNT_DRAWINGS_POS_CURRENT_mean': [1.5],
            'CNT_INSTALMENT_MATURE_CUM_max': [26],
            'CNT_INSTALMENT_MATURE_CUM_min': [26],
            'CNT_INSTALMENT_MATURE_CUM_mean': [26.0],
            'SK_DPD_max': [5],
            'SK_DPD_min': [2],
            'SK_DPD_mean': [3.5],
            'SK_DPD_DEF_max': [3],
            'SK_DPD_DEF_min': [1],
            'SK_DPD_DEF_mean': [2.0],
            'NAME_CONTRACT_STATUS_Active_sum': [2],
        })    
        pd.testing.assert_frame_equal(result, expected_result)

    def test_get_aggregated_installments_payments_data(self):
        # Arrange
        installments_payments_data = pd.DataFrame({
            "SK_ID_PREV": [1903401, 1915321],
            "SK_ID_CURR": [100267, 100267],
            "NUM_INSTALMENT_VERSION": [1, 2],
            "NUM_INSTALMENT_NUMBER": [72, 2],
            "DAYS_INSTALMENT": [-1212, -62],
            "DAYS_ENTRY_PAYMENT": [-1210, -60],
            "AMT_INSTALMENT": [250.0, 13500.0],
            "AMT_PAYMENT": [250.0, 13500.0]
        })

        # Act
        result = self.reader.get_aggregated_installments_payments_data(installments_payments_data)

        # Assert
        expected_result = pd.DataFrame({
            'SK_ID_CURR': [100267],
            'NUM_INSTALMENT_VERSION_max': [2],
            'NUM_INSTALMENT_VERSION_min': [1],
            'NUM_INSTALMENT_VERSION_mean': [1.5],
            'NUM_INSTALMENT_NUMBER_max': [72],
            'NUM_INSTALMENT_NUMBER_min': [2],
            'NUM_INSTALMENT_NUMBER_mean': [37.0],
            'DAYS_INSTALMENT_max': [-62],
            'DAYS_INSTALMENT_min': [-1212],
            'DAYS_ENTRY_PAYMENT_max': [-60],
            'DAYS_ENTRY_PAYMENT_min': [-1210],
            'AMT_INSTALMENT_max': [13500.0],
            'AMT_INSTALMENT_min': [250.0],
            'AMT_INSTALMENT_mean': [6875.0],
            'AMT_PAYMENT_max': [13500.0],
            'AMT_PAYMENT_min': [250.0],
            'AMT_PAYMENT_mean': [6875.0],
        })
        pd.testing.assert_frame_equal(result, expected_result)

    def test_get_aggregated_previous_application_data(self):
        # Arrange
        previous_application_data = pd.DataFrame({
            "SK_ID_PREV": [2670402, 2562193],
            "SK_ID_CURR": [100077, 100077],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
            "AMT_ANNUITY": [5000, 2250],
            "AMT_APPLICATION": [100000, 45000],
            "AMT_CREDIT": [120000, 45000],
            "AMT_DOWN_PAYMENT": [20000, 5000],
            "AMT_GOODS_PRICE": [80000, 45000],
            "WEEKDAY_APPR_PROCESS_START": ["WEDNESDAY", "WEDNESDAY"],
            "HOUR_APPR_PROCESS_START": [14, 15],
            "FLAG_LAST_APPL_PER_CONTRACT": ["Y", "Y"],
            "NFLAG_LAST_APPL_IN_DAY": [1, 1],
            "NFLAG_MICRO_CASH": [0, 0],
            "RATE_DOWN_PAYMENT": [0.25, 0.1],
            "RATE_INTEREST_PRIMARY": [0.03, 0.05],
            "RATE_INTEREST_PRIVILEGED": [0.02, 0.04],
            "NAME_CASH_LOAN_PURPOSE": ["XNA", "XAP"],
            "NAME_CONTRACT_STATUS": ["Refused", "Approved"],
            "DAYS_DECISION": [-217, -217],
            "NAME_PAYMENT_TYPE": ["XNA", "XNA"],
            "CODE_REJECT_REASON": ["HC", "XAP"],
            "NAME_TYPE_SUITE": ["Unaccompanied", "Unaccompanied"],
            "NAME_CLIENT_TYPE": ["Repeater", "Repeater"],
            "NAME_GOODS_CATEGORY": ["XNA", "XNA"],
            "NAME_PORTFOLIO": ["XNA", "Cards"],
            "NAME_PRODUCT_TYPE": ["XNA", "walk-in"],
            "CHANNEL_TYPE": ["Credit and cash offices", "Credit and cash offices"],
            "SELLERPLACE_AREA": [-1, -1],
            "NAME_SELLER_INDUSTRY": ["XNA", "XNA"],
            "CNT_PAYMENT": [24, 12],
            "NAME_YIELD_GROUP": ["XNA", "XNA"],
            "PRODUCT_COMBINATION": ["Cash", "Card Street"],
            "DAYS_FIRST_DRAWING": [None, None],
            "DAYS_FIRST_DUE": [-365, -30],
            "DAYS_LAST_DUE_1ST_VERSION": [-300, -15],
            "DAYS_LAST_DUE": [-150, -5],
            "DAYS_TERMINATION": [-100, -1],
            "NFLAG_INSURED_ON_APPROVAL": [0, 1],
        })

        # Act
        result = self.reader.get_aggregated_previous_application_data(previous_application_data)

        # Assert
        expected_result = pd.DataFrame({
            'SK_ID_CURR': [100077],
            'AMT_ANNUITY_max': [5000],
            'AMT_ANNUITY_min': [2250],
            'AMT_ANNUITY_mean': [3625.0],
            'AMT_APPLICATION_max': [100000],
            'AMT_APPLICATION_min': [45000],
            'AMT_APPLICATION_mean': [72500.0],
            'AMT_CREDIT_max': [120000],
            'AMT_CREDIT_min': [45000],
            'AMT_CREDIT_mean': [82500.0],
            'AMT_DOWN_PAYMENT_max': [20000],
            'AMT_DOWN_PAYMENT_min': [5000],
            'AMT_DOWN_PAYMENT_mean': [12500.0],
            'AMT_GOODS_PRICE_max': [80000],
            'AMT_GOODS_PRICE_min': [45000],
            'AMT_GOODS_PRICE_mean': [62500.0],
            'HOUR_APPR_PROCESS_START_max': [15],
            'HOUR_APPR_PROCESS_START_min': [14],
            'HOUR_APPR_PROCESS_START_mean': [14.5],
            'RATE_DOWN_PAYMENT_max': [0.25],
            'RATE_DOWN_PAYMENT_min': [0.1],
            'RATE_DOWN_PAYMENT_mean': [0.175],
            'RATE_INTEREST_PRIMARY_max': [0.05],
            'RATE_INTEREST_PRIMARY_min': [0.03],
            'RATE_INTEREST_PRIMARY_mean': [0.04],
            'RATE_INTEREST_PRIVILEGED_max': [0.04],
            'RATE_INTEREST_PRIVILEGED_min': [0.02],
            'RATE_INTEREST_PRIVILEGED_mean': [0.03],
            'DAYS_DECISION_max': [-217],
            'DAYS_DECISION_min': [-217],
            'DAYS_DECISION_mean': [-217.0],
            'SELLERPLACE_AREA_max': [-1],
            'SELLERPLACE_AREA_min': [-1],
            'SELLERPLACE_AREA_mean': [-1.0],
            'CNT_PAYMENT_max': [24],
            'CNT_PAYMENT_min': [12],
            'CNT_PAYMENT_mean': [18.0],
            'DAYS_FIRST_DRAWING_max': [None],
            'DAYS_FIRST_DRAWING_min': [None],
            'DAYS_FIRST_DRAWING_mean': [None],
            'DAYS_FIRST_DUE_max': [-30],
            'DAYS_FIRST_DUE_min': [-365],
            'DAYS_FIRST_DUE_mean': [-197.5],
            'DAYS_LAST_DUE_1ST_VERSION_max': [-15],
            'DAYS_LAST_DUE_1ST_VERSION_min': [-300],
            'DAYS_LAST_DUE_1ST_VERSION_mean': [-157.5],
            'DAYS_LAST_DUE_max': [-5],
            'DAYS_LAST_DUE_min': [-150],
            'DAYS_LAST_DUE_mean': [-77.5],
            'DAYS_TERMINATION_max': [-1],
            'DAYS_TERMINATION_min': [-100],
            'DAYS_TERMINATION_mean': [-50.5],
            'NAME_CONTRACT_TYPE_Cash loans_sum': [1],
            'NAME_CONTRACT_TYPE_Revolving loans_sum': [1],
            'WEEKDAY_APPR_PROCESS_START_WEDNESDAY_sum': [2],
            'FLAG_LAST_APPL_PER_CONTRACT_Y_sum': [2],
            'NFLAG_LAST_APPL_IN_DAY_1_sum': [2],
            'NAME_CASH_LOAN_PURPOSE_XAP_sum': [1],
            'NAME_CASH_LOAN_PURPOSE_XNA_sum': [1],
            'NAME_CONTRACT_STATUS_Approved_sum': [1],
            'NAME_CONTRACT_STATUS_Refused_sum': [1],
            'NAME_PAYMENT_TYPE_XNA_sum': [2],
            'CODE_REJECT_REASON_HC_sum': [1],
            'CODE_REJECT_REASON_XAP_sum': [1],
            'NAME_TYPE_SUITE_Unaccompanied_sum': [2],
            'NAME_CLIENT_TYPE_Repeater_sum': [2],
            'NAME_GOODS_CATEGORY_XNA_sum': [2],
            'NAME_PORTFOLIO_Cards_sum': [1],
            'NAME_PORTFOLIO_XNA_sum': [1],
            'NAME_PRODUCT_TYPE_XNA_sum': [1],
            'NAME_PRODUCT_TYPE_walk-in_sum': [1],
            'CHANNEL_TYPE_Credit and cash offices_sum': [2],
            'NAME_SELLER_INDUSTRY_XNA_sum': [2],
            'NAME_YIELD_GROUP_XNA_sum': [2],
            'PRODUCT_COMBINATION_Card Street_sum': [1],
            'PRODUCT_COMBINATION_Cash_sum': [1],
            'NFLAG_INSURED_ON_APPROVAL_0_sum': [1],
            'NFLAG_INSURED_ON_APPROVAL_1_sum': [1],
        })
        pd.testing.assert_frame_equal(result, expected_result)

    def test_get_aggregated_pos_cash_balance_data(self):
        # Arrange
        pos_cash_balance_data = pd.DataFrame({
            "SK_ID_PREV": [2664977, 1143677],
            "SK_ID_CURR": [100187, 100187],
            "MONTHS_BALANCE": [-43, -12],
            "CNT_INSTALMENT": [24, 60],
            "CNT_INSTALMENT_FUTURE": [8, 59],
            "NAME_CONTRACT_STATUS": ["Active", "Active"],
            "SK_DPD": [0, 0],
            "SK_DPD_DEF": [0, 0]
        })

        # Act
        result = self.reader.get_aggregated_pos_cash_balance_data(pos_cash_balance_data)

        # Assert
        expected_result = pd.DataFrame({
            'SK_ID_CURR': [100187],
            'MONTHS_BALANCE_max': [-12],
            'MONTHS_BALANCE_min': [-43],
            'CNT_INSTALMENT_max': [60],
            'CNT_INSTALMENT_min': [24],
            'CNT_INSTALMENT_mean': [42.0],
            'CNT_INSTALMENT_FUTURE_max': [59],
            'CNT_INSTALMENT_FUTURE_min': [8],
            'CNT_INSTALMENT_FUTURE_mean': [33.5],
            'SK_DPD_max': [0],
            'SK_DPD_min': [0],
            'SK_DPD_mean': [0.0],
            'SK_DPD_DEF_max': [0],
            'SK_DPD_DEF_min': [0],
            'SK_DPD_DEF_mean': [0.0],
            'NAME_CONTRACT_STATUS_Active_sum': [2],
        }) 
        pd.testing.assert_frame_equal(result, expected_result)

    @patch('pandas.read_csv')
    @patch('backend.src.data_processing.simple_read_data.SimpleReadData.get_aggregated_bureau_data')
    @patch('backend.src.data_processing.simple_read_data.SimpleReadData.get_aggregated_credit_card_balance_data')
    @patch('backend.src.data_processing.simple_read_data.SimpleReadData.get_aggregated_installments_payments_data')
    @patch('backend.src.data_processing.simple_read_data.SimpleReadData.get_aggregated_previous_application_data')
    @patch('backend.src.data_processing.simple_read_data.SimpleReadData.get_aggregated_pos_cash_balance_data')
    def test_retrieve_data_concat(self, mock_pos, mock_previous, mock_installments, mock_credit, mock_bureau, mock_read_csv):
        # Create a mock DataFrame to return from pd.read_csv
        mock_df = pd.DataFrame({
            'SK_ID_CURR': [1, 2, 3],
            'DATA': ['A', 'B', 'C']
        })
        mock_read_csv.return_value = mock_df

        # Create a mock DataFrame to return from the get_aggregated_* methods
        mock_aggregated_bureau = pd.DataFrame({
            'SK_ID_CURR': [1, 2, 3],
            'BUREAU_AGGREGATED_DATA': ['D', 'E', 'F']
        })
        mock_aggregated_credit = pd.DataFrame({
            'SK_ID_CURR': [1, 2, 3],
            'CREDIT_AGGREGATED_DATA': ['D', 'E', 'F']
        })
        mock_aggregated_installments = pd.DataFrame({
            'SK_ID_CURR': [1, 2, 3],
            'INSTALLMENTS_AGGREGATED_DATA': ['D', 'E', 'F']
        })
        mock_aggregated_previous = pd.DataFrame({
            'SK_ID_CURR': [1, 2, 3],
            'PREVIOUS_AGGREGATED_DATA': ['D', 'E', 'F']
        })
        mock_aggregated_pos = pd.DataFrame({
            'SK_ID_CURR': [1, 2, 3],
            'POS_AGGREGATED_DATA': ['D', 'E', 'F']
        })

        mock_bureau.return_value = mock_aggregated_bureau
        mock_credit.return_value = mock_aggregated_credit
        mock_installments.return_value =  mock_aggregated_installments
        mock_previous.return_value = mock_aggregated_previous
        mock_pos.return_value = mock_aggregated_pos

        # Create an instance of the class and call the method
        result = self.reader.retrieve_data('mock_path', 2)

        # Check that the result is as expected
        expected_result = pd.DataFrame({
            'SK_ID_CURR': [1, 2, 3],
            'DATA': ['A', 'B', 'C'],
            'BUREAU_AGGREGATED_DATA': ['D', 'E', 'F'],
            'CREDIT_AGGREGATED_DATA': ['D', 'E', 'F'],
            'INSTALLMENTS_AGGREGATED_DATA': ['D', 'E', 'F'],
            'PREVIOUS_AGGREGATED_DATA': ['D', 'E', 'F'],
            'POS_AGGREGATED_DATA': ['D', 'E', 'F'],
        })

        self.assertTrue(isinstance(result, pd.DataFrame))
        pd.testing.assert_frame_equal(result, expected_result)

    @patch('pandas.DataFrame.to_csv')
    @patch('backend.src.data_processing.simple_read_data.SimpleReadData.retrieve_data')
    def test_write_data(self, mock_read_data, mock_to_csv):
        # Create a mock DataFrame to return from read_data
        mock_df = pd.DataFrame({
            'SK_ID_CURR': [1, 2, 3],
            'DATA': ['A', 'B', 'C']
        })
        mock_read_data.return_value = mock_df

        mock_path = 'mock_path'
        mock_file = 'mock_file'

        # Create an instance of the class and call the method
        self.reader.write_data(mock_path, mock_file)

        # Check that the result is as expected
        mock_read_data.assert_called_once_with(mock_path, sampling_frequency = 1, training = True)
        mock_df.to_csv.assert_called_once_with(f"{mock_path}/{mock_file}", index=False)

    @patch('pandas.read_csv')
    def test_read_data(self, mock_read_csv):
        # Arrange
        mock_df = pd.DataFrame({
            'SK_ID_CURR': [1, 2, 3],
            'DATA': ['A', 'B', 'C']
        })
        mock_read_csv.return_value = mock_df

        mock_path = 'mock_path'
        mock_file = 'mock_file'

        # Act
        result = self.reader.read_data(mock_path, mock_file)

        # Assert
        mock_read_csv.assert_called_once_with(f"{mock_path}/{mock_file}")
        self.assertTrue(isinstance(result, pd.DataFrame))
        pd.testing.assert_frame_equal(result, mock_df)

    @patch('builtins.open')
    @patch('json.dump')
    def test_write_data_structure_json(self, mock_json_dump, mock_open):
        # Arrange
        mock_df = pd.DataFrame({
            'SK_ID_CURR': [1, 2, 3],
            'LOAN_AMOUNT': [100000.5, 200000.5, 300000.5],
            'DATA': ['A', 'B', 'C'],
            'FLAG': [1, 0, 1]
        })
        mock_path = 'mock_path'
        mock_file = 'mock_file.json'

        mock_open.return_value.__enter__.return_value = mock_open
        mock_open.return_value.__exit__.return_value = None

        # Act
        self.reader.write_data_structure_json(mock_df, mock_path, mock_file)

        # Assert
        mock_open.assert_called_once_with(f"{mock_path}/{mock_file}", 'w')
        mock_json_dump.assert_called_once_with({
            'SK_ID_CURR': {
                'type': 'int64',
                'min': 1,
                'max': 3
            },
            'LOAN_AMOUNT': {
                'type': 'float64',
                'min': 100000,
                'max': 300000
            },
            'DATA': {
                'type': 'object',
                'values': ['A', 'B', 'C']
            },
            'FLAG': {
                'type': 'int64',
                'values': [1, 0]
            }
        }, mock_open, indent=4)

if __name__ == '__main__':
    unittest.main()