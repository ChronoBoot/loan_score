import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd

from .loan_predictor_abc import LoanPredictor

class RandomForestLoanPredictor(LoanPredictor):
    """
    A concrete implementation of the LoanPredictor abstract base class using a Random Forest Classifier.

    This class provides functionality to train a Random Forest model on loan data, predict the outcome of a loan,
    and evaluate the performance of the model.
    """

    def __init__(self) -> None:
        self.model = RandomForestClassifier()
        self.label_encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.random_state = 42
        self.test_size = 0.2
        logging.debug("RandomForestLoanPredictor initialized")

    def preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data by encoding categorical variables and filling NaN values.

        Args:
            X (pd.DataFrame): The DataFrame to preprocess. It is not modified.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """

        new_data = X.copy()

        for column in new_data.columns:
            if new_data[column].dtype == 'object':
                le = LabelEncoder()
                le.fit(new_data[column])
                self.label_encoders[column] = le
                new_data[column] = le.transform(new_data[column])

        
        # Fill NaN values
        new_data = new_data.fillna(0)

        logging.debug("Data preprocessed")

        return new_data

    def train(self, loans: pd.DataFrame, target_variable: str) -> None:
        """
        Train the predictor on a DataFrame of loans.

        Args:
            loans (pd.DataFrame): The DataFrame of loans to train the predictor on.
            target_variable (str): The name of the target variable in the DataFrame.

        Returns:
            None
        """
        try:
            # Drop the target variable from the training data
            X = loans.drop(columns=[target_variable])
            X = self.preprocess_data(X)

            y = loans[target_variable]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

            self.model.fit(self.X_train, self.y_train)

            logging.debug("Model trained")
        except Exception as e:
            logging.error(f"Failed to train the model: {e}")

    def evaluate(self) -> float:
        """
        Evaluate the performance of the predictor.

        Returns:
            float: The accuracy of the model.
        """
        try:
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            logging.debug(f"Model evaluated with accuracy: {accuracy}")
            return accuracy
        except Exception as e:
            logging.error(f"Failed to evaluate the model: {e}")

    def predict(self, loan: pd.DataFrame) -> int:
        """
        Predict the outcome for a loan.

        Args:
            loan (pd.DataFrame): The loan to predict the outcome for.

        Returns:
            int: The predicted outcome for the loan. 1 for a rejected loan, 0 for an accepted loan.
        """
        try:
            ordered_loan = loan[self.X_train.columns]

            for column in ordered_loan.columns:
                if ordered_loan[column].dtype == 'object':
                    ordered_loan[column] = self.label_encoders[column].transform(ordered_loan[column])

            prediction = int(self.model.predict(ordered_loan))

            logging.debug(f"Predicted outcome for loan: {prediction}")

            return prediction
        except Exception as e:
            logging.error(f"Failed to predict the outcome for the loan: {e}")

    def get_most_important_features(self, nb_features : int) -> pd.DataFrame:
        """
        Get the most important features from the model.

        Returns:
            pd.DataFrame: A DataFrame of the most important features.
        """
        try:
            feature_importances = pd.DataFrame(self.model.feature_importances_, index = self.X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
            logging.debug(f"Got the most important features")
            return feature_importances.head(nb_features)
        except Exception as e:
            logging.error(f"Failed to get the most important features: {e}")