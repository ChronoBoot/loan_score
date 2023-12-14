from sklearn.base import accuracy_score
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from loan_predictor_abc import LoanPredictor

class RandomForestLoanPredictor(LoanPredictor):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self, loans):
        target_variable = 'TARGET'

        # Drop the target variable from the training data
        X = loans.drop(columns=[target_variable])
        
        # Initialize a LabelEncoder
        le = LabelEncoder()

        # Apply the LabelEncoder to each column
        X = X.apply(lambda col: le.fit_transform(col) if col.dtype == 'object' else col)

        # Fill NaN values
        X = X.fillna(0)

        y = loans[target_variable]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy