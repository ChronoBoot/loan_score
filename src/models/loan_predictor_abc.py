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
            int: The predicted outcome for the loan. 0 for a rejected loan, 1 for an accepted loan.
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