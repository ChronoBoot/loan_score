import logging
from frontend.src.ui.user_interface_abc import UserInterface
from dash import Dash, html, Input, Output, State, dcc, callback, no_update
from backend.src.models.random_forest_loan_predictor import RandomForestLoanPredictor
import os
import pandas as pd
import requests
import json

class DashUserInterface(UserInterface):
    """
    A class representing the user interface for loan prediction using Dash framework.

    Args:
        model (RandomForestLoanPredictor): The loan prediction model.
        categorical_values (dict): A dictionary containing the categorical feature names as keys and their possible values as values.
        float_values (list): A list of float feature names.

    Attributes:
        app (Dash): The Dash application instance.
        model (RandomForestLoanPredictor): The loan prediction model.
        categorical_values (dict): A dictionary containing the categorical feature names as keys and their possible values as values.
        float_values (list): A list of float feature names.

    Methods:
        display(): Starts the Dash server and displays the user interface.
        predict(data=[]): Predicts the loan using the given data.

    """

    NB_VALUES_SLIDER = 100
    SERVER_PORT = 11000

    def __init__(self, categorical_values : dict, float_values: dict, loan_example: dict, field_descriptions: dict, predict_url: str) -> None: 
        self.app = Dash(__name__)
        self.original_categorical_values = categorical_values
        self.categorical_values = self.update_categorical_values(categorical_values)
        self.float_values = float_values
        self.loan_example = loan_example
        self.field_descriptions = field_descriptions
        self.predict_url = predict_url

        self.app.callback(
            Output('prediction-popup', 'displayed'),
            Output('prediction-popup', 'message'),
            Input('predict-button', 'n_clicks'),
            [State(dropdown, 'value') for dropdown in self.categorical_values.keys()],
            [State(input, 'value') for input in self.float_values.keys()]
        )(self._update_callback)

        self.app.callback(
            Output('prediction', 'children'),
            Input('predict-button', 'n_clicks')
        )(self._clear_prediction_callback)

        self.app.layout = self._create_layout()

        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        logging.basicConfig(format=log_format, level=logging.DEBUG)

    def get_nb_steps(self, min, max) -> int:
        """
        Gets the number of steps for the user interface slider.

        Returns:
            The number of steps.
        """
        diff = max - min

        if diff <= 1:
            return diff/DashUserInterface.NB_VALUES_SLIDER
        elif diff <= DashUserInterface.NB_VALUES_SLIDER:
            return 1
        else:
            return round(diff/DashUserInterface.NB_VALUES_SLIDER)

    def update_categorical_values(self, categorical_values: dict) -> dict:
        """
        Updates the categorical values from the backend.
        Booleans values are rewritten to Yes/No.
        """
        updated_categorical_values = {}
        for column, values in categorical_values.items():
            if set(values) == {0, 1} or set(values) == {'Y', 'N'}:
                updated_categorical_values[column] = ['Yes', 'No']
            else:
                updated_categorical_values[column] = values
        return updated_categorical_values
    
    def format_booleans(self, data: dict) -> dict:
        """
        Formats the boolean values to 0/1.
        """
        for column, values in data.items():
            if column in self.original_categorical_values and set(self.original_categorical_values[column]) == {0, 1}:
                data[column] = 1 if values == 'Yes' else 0
            elif column in self.original_categorical_values and set(self.original_categorical_values[column]) == {'Y', 'N'}:
                data[column] = '1' if values == 'Yes' else '0'
        return data

    def _create_layout(self):
        """
        Creates the layout for the Dash application.

        Returns:
            The Dash application layout.
        """
        layout = html.Div([
            html.H1(
                children='Loan prediction',
                id='prediction',
                style={'textAlign': 'center'}
            ),
            *[html.Div([
                html.Label(self.field_descriptions[col]),
                dcc.Dropdown(
                    id=col, 
                    options=[{'label': val, 'value': val} for val in values],
                    placeholder=f"Select {col}",
                    value=self.loan_example[col]
                )
            ]) for col, values in self.categorical_values.items()],
            *[html.Div([
                html.Label(self.field_descriptions[col]),
                dcc.Slider(
                    id=col, 
                    min=self.float_values[col]['min'],
                    max=self.float_values[col]['max'],
                    step=self.get_nb_steps(self.float_values[col]['min'], self.float_values[col]['max']),
                    # Because some fields are aggregated from the values, some might be missing
                    value=self.loan_example[col] if col in self.loan_example else 0,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ]) for col in self.float_values.keys()],
            html.Button('Predict', id='predict-button', n_clicks=0),
            dcc.ConfirmDialog(
                id='prediction-popup',
                message='',
                displayed=False
            )
        ])

        return layout

    def _update_callback(self, n_clicks, *args):
        """
        Callback function for updating the prediction popup.

        Args:
            n_clicks (int): The number of times the predict button has been clicked.
            *args: The values of the dropdowns and inputs.

        Returns:
            The updated state of the prediction popup.
        """
        if n_clicks > 0:
            combined_keys = list(self.categorical_values.keys()) + list(self.float_values.keys())
            data = dict(zip(combined_keys, args))
            data = self.format_booleans(data)

            try:            
                prediction = self.predict(data)

                if(prediction == 1):
                    return True, 'The prediction is: Loan will not be repaid'
                else:
                    return True, 'The prediction is: Loan will be repaid'
            except ValueError:
                return True, 'The prediction went wrong. Please try again.'

        return no_update, ''

    def _clear_prediction_callback(self, n_clicks):
        """
        Callback function for clearing the prediction.

        Args:
            n_clicks (int): The number of times the predict button has been clicked.

        Returns:
            The updated state of the prediction.
        """
        if n_clicks > 0:
            return ''
        return no_update

    def display(self):
        """
        Starts the Dash server and displays the user interface.
        """
        port = int(os.environ.get("PORT", DashUserInterface.SERVER_PORT))
        host = os.getenv("HOST", '0.0.0.0')
        self.app.run_server(debug=True, host=host, port=port)

    def predict(self, data : dict =[]) -> int:
        """
        Predicts the loan using the given data.

        Args:
            data (list, optional): The data to be used for prediction. Defaults to an empty list.

        Returns:
            The loan prediction.
        """
        json_dict = {
            'loan' : data
        }

        json_data = json.dumps(json_dict)
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self.predict_url, data=json_data, headers=headers)
        
        if response.status_code == 200:
            prediction = response.json()
            logging.debug(f"Predicted outcome for loan: {prediction}")
            if(prediction['prediction'] == None):
                raise ValueError('Prediction is None. Something went wrong with the model')
            return int(prediction['prediction'])
        else:
            return None
