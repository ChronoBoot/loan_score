import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from unittest.mock import Mock, patch, MagicMock
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from src.models.random_forest_loan_predictor import RandomForestLoanPredictor

class TestRandomForestLoanPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = RandomForestLoanPredictor()

    def test_init(self):
        self.assertIsInstance(self.predictor.model, RandomForestClassifier)
        self.assertEqual(self.predictor.label_encoders, {})

    @patch('pandas.DataFrame.fillna')
    @patch('pandas.DataFrame.copy')
    @patch('src.models.random_forest_loan_predictor.LabelEncoder')
    def test_preprocess_data(self, mock_label_encoder, mock_copy, mock_fillna):
        # Define some constants for the test
        default_value = 0
        col1_name = 'col1'
        col2_name = 'col2'

        # Define the expected result of the preprocessing
        expected_result = pd.DataFrame({
            col1_name: [default_value, default_value, default_value],
            col2_name: [default_value, default_value, default_value],
        })

        # Define the input DataFrame
        test_df = pd.DataFrame({
            col1_name: ['a', 'b', 'c'],
            col2_name: ['toto', 'tata', 'titi'],
        })

        # Define a copy of the input DataFrame
        copy_df = pd.DataFrame({
            col1_name: ['a', 'b', 'c'],
            col2_name: ['toto', 'tata', 'titi'],
        })

        # Set up the mock for fillna to return the copy
        mock_fillna.return_value = copy_df
        copy_df.fillna = mock_fillna

        # Set up the mock for copy to return the copy
        mock_copy.return_value = copy_df
        test_df.copy = mock_copy

        # Set up the mock for LabelEncoder
        mock_label_encoder_instance = mock_label_encoder.return_value
        mock_label_encoder_instance.transform.return_value = default_value
        mock_label_encoder_instance.fit.return_value = default_value

        # Call the method under test
        result = self.predictor.preprocess_data(test_df)
        
        # Assert that copy was called once
        test_df.copy.assert_called_once()

        # Assert that fillna was called once with 0 as argument
        copy_df.fillna.assert_called_once()
        copy_df.fillna.assert_called_with(0)

        # Check the arguments of the first call to fit
        args, _ = mock_label_encoder_instance.fit.call_args_list[0]
        assert_series_equal(args[0], test_df[col1_name])

        # Check the arguments of the second call to fit
        args, _ = mock_label_encoder_instance.fit.call_args_list[1]
        assert_series_equal(args[0], test_df[col2_name])

        # Check the arguments of the first call to transform
        args, _ = mock_label_encoder_instance.transform.call_args_list[0]
        assert_series_equal(args[0], test_df[col1_name])

        # Check the arguments of the second call to transform
        args, _ = mock_label_encoder_instance.transform.call_args_list[1]
        assert_series_equal(args[0], test_df[col2_name])

        # Assert that the result is as expected
        assert_frame_equal(result, expected_result)

    @patch('pandas.DataFrame.drop')
    @patch('src.models.random_forest_loan_predictor.RandomForestLoanPredictor.preprocess_data')
    @patch('src.models.random_forest_loan_predictor.train_test_split')
    @patch('src.models.random_forest_loan_predictor.RandomForestClassifier.fit')
    def test_train(self, mock_fit, mock_split, mock_preprocess, mock_drop):
        # Define some constants for the test
        col1_name = 'col1'
        target_col_name = 'target'

        # Define the input DataFrame
        test_df = pd.DataFrame({
            col1_name: ['a', 'b', 'c'],
            target_col_name: [1, 0, 1],
        })

        # Define the DataFrame that should be the result of dropping the target column
        test_df_without_target = pd.DataFrame({
            col1_name: ['a', 'b', 'c'],
        })

        # Define the DataFrame that should be the result of preprocessing
        test_df_after_preprocessing = pd.DataFrame({
            col1_name: [0, 1, 2]
        })

        # Define the result of train_test_split
        train_test_split_result = [test_df, test_df, test_df, test_df]

        # Set up the mock for drop to return the DataFrame without the target column
        mock_drop.return_value = test_df_without_target
        test_df.drop = mock_drop

        # Set up the mock for preprocess_data to return the preprocessed DataFrame
        mock_preprocess.return_value = test_df_after_preprocessing
        self.predictor.preprocess_data = mock_preprocess

        # Set up the mock for train_test_split to return the predefined result
        mock_split.return_value = train_test_split_result

        # Set up the mock for fit to return a mock RandomForestClassifier
        mock_fit.return_value = Mock(spec=RandomForestClassifier)
        self.predictor.model.fit = mock_fit

        # Call the method under test
        self.predictor.train(test_df, target_col_name)

        # Assert that drop was called once with the target column as argument
        mock_drop.assert_called_once()
        mock_drop.assert_called_with(columns=[target_col_name])

        # Assert that preprocess_data was called once with the DataFrame without the target column as argument
        mock_preprocess.assert_called_once()
        mock_preprocess.assert_called_with(test_df_without_target)

        # Assert that train_test_split was called once with the preprocessed DataFrame and the target column as arguments
        mock_split.assert_called_once()
        args, kwargs = mock_split.call_args_list[0]
        assert_frame_equal(args[0], test_df_after_preprocessing)
        assert_series_equal(args[1], test_df[target_col_name])
        self.assertEqual(kwargs['test_size'], self.predictor.test_size)
        self.assertEqual(kwargs['random_state'], self.predictor.random_state)

        # Assert that fit was called once with the training data and target variable
        mock_fit.assert_called_once()
        mock_fit.assert_called_with(test_df, self.predictor.y_train)


    @patch('src.models.random_forest_loan_predictor.RandomForestClassifier.predict')
    @patch('src.models.random_forest_loan_predictor.accuracy_score')
    def test_evaluate(self, mock_accuracy_score, mock_predict):
        # Define a constant for the accuracy
        accuracy = 0.9

        # Set up the mock for accuracy_score to return the predefined accuracy
        mock_accuracy_score.return_value = accuracy

        # Define a list for the predictions
        predictions = [1, 0, 1]

        # Set up the mock for predict to return the predefined predictions
        mock_predict.return_value = predictions
        self.predictor.model.predict = mock_predict

        # Call the method under test
        result = self.predictor.evaluate()

        # Assert that predict was called once with the test data as argument
        mock_predict.assert_called_once()
        mock_predict.assert_called_with(self.predictor.X_test)

        # Assert that accuracy_score was called once with the test target variable and the predictions as arguments
        mock_accuracy_score.assert_called_once()
        mock_accuracy_score.assert_called_with(self.predictor.y_test, predictions)

        # Assert that the result is as expected
        self.assertEqual(result, accuracy)

    @patch('src.models.random_forest_loan_predictor.LabelEncoder.transform')
    @patch('src.models.random_forest_loan_predictor.RandomForestClassifier.predict')
    def test_predict(self, mock_predict, mock_transform):
        # Define some constants for the test
        prediction_value = 1
        transform_value = 0
        col1_name = 'col1'
        col2_name = 'col2'

        # Define the input DataFrame
        test_df = pd.DataFrame({
            'col1': ['a', 'b', 'c'],
            'col2': ['toto', 'tata', 'titi'],
        })

        self.predictor.X_train = pd.DataFrame({
                'col1': [0, 1, 2],
                'col2': [0, 1, 2],
            })

        # Define the DataFrame that should be the result of transforming the input DataFrame
        order_loan = pd.DataFrame({
            col1_name: [transform_value, transform_value, transform_value],
            col2_name: [transform_value, transform_value, transform_value],
        })

        # Set up the mock for predict to return the predefined prediction value
        mock_predict.return_value = prediction_value
        self.predictor.model.predict = mock_predict

        # Set up the mock for transform to return the predefined transform value
        mock_transform.return_value = transform_value
        self.predictor.label_encoders['col1'] = Mock(spec=LabelEncoder)
        self.predictor.label_encoders['col2'] = Mock(spec=LabelEncoder)
        self.predictor.label_encoders['col1'].transform = mock_transform
        self.predictor.label_encoders['col2'].transform = mock_transform

        # Call the method under test
        result = self.predictor.predict(test_df)

        # Assert that transform was called once with the first column of the input DataFrame as argument
        args, _ = mock_transform.call_args_list[0]
        assert_series_equal(args[0], test_df['col1'])        

        # Assert that transform was called once with the second column of the input DataFrame as argument
        args, _ = mock_transform.call_args_list[1]
        assert_series_equal(args[0], test_df['col2']) 

        # Assert that predict was called once with the transformed DataFrame as argument
        mock_predict.assert_called_once()
        args, _ = mock_predict.call_args_list[0]
        assert_frame_equal(args[0], order_loan)

        # Assert that the result is as expected
        self.assertEqual(result, prediction_value)

    @patch('src.models.random_forest_loan_predictor.RandomForestClassifier')
    def test_get_most_important_features(self, mock_model):
        # Define some constants for the test
        nb_features = 2
        columns = ['col1', 'col2', 'col3']
        importances = [0.2, 0.3, 0.5]

        # Define the input DataFrame
        X_train = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=columns)

        # Define the expected output DataFrame
        expected_output = pd.DataFrame(importances, index=columns, columns=['importance']).sort_values('importance', ascending=False).head(nb_features)

        # Set up the mock for the model's feature_importances_ attribute
        mock_model.feature_importances_ = importances

        # Create an instance of RandomForestLoanPredictor
        predictor = RandomForestLoanPredictor()
        predictor.model = mock_model
        predictor.X_train = X_train

        # Call the method under test
        result = predictor.get_most_important_features(nb_features)

        # Assert that the result is as expected
        pd.testing.assert_frame_equal(result, expected_output)

if __name__ == '__main__':
    unittest.main()