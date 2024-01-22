from abc import ABC, abstractmethod
import pandas as pd

class LoanPredictor(ABC):
    """
    Abstract base class that defines a contract for a loan predictor.

    Any class that inherits from this must implement the `predict`, `train`, and `evaluate` methods.
    """

    @abstractmethod
    def predict(self, loan: pd.DataFrame) -> int:
        """
        Predict the outcome for a loan.

        Args:
            loan (pd.DataFrame): The loan to predict the outcome for.

        Returns:
            int: The predicted outcome for the loan. 1 for a rejected loan, 0 for an accepted loan.
        """
        pass

    @abstractmethod
    def train(self, loans: pd.DataFrame) -> None:
        """
        Train the predictor on a DataFrame of loans.

        Args:
            loans (pd.DataFrame): The DataFrame of loans to train the predictor on.

        Returns:
            None
        """
        pass

    @abstractmethod
    def evaluate(self) -> float:
        """
        Evaluate the performance of the predictor.

        Returns:
            float: The result of the evaluation.
        """
        pass

    @abstractmethod
    def get_most_important_features(self, nb_features : int) -> pd.DataFrame:
        """
        Get the most important features from the model.

        Returns:
            pd.DataFrame: A DataFrame of the most important features.
        """
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Save the model to a file.

        Args:
            path (str): The path to save the model to.

        Returns:
            None
        """
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Load the model from a file.

        Args:
            path (str): The path to load the model from.

        Returns:
            None
        """
        pass