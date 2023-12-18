from abc import ABC, abstractmethod

class UserInterface(ABC):
    """
    Abstract base class for a user interface.
    """

    @abstractmethod
    def display(self):
        """
        Display the user interface.
        This method should be implemented by subclasses.
        """
        pass