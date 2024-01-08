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

    API_URL = "http://127.0.0.1:5000"
    PREDICT_URL = f"{API_URL}/predict"

    def __init__(self, categorical_values : dict, float_values: dict) -> None: 
        self.app = Dash(__name__)
        self.categorical_values = categorical_values
        self.float_values = float_values

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
                html.Label(col),
                dcc.Dropdown(
                    id=col, 
                    options=[{'label': val, 'value': val} for val in values],
                    placeholder=f"Select {col}",
                    value=values[0]
                )
            ]) for col, values in self.categorical_values.items()],
            *[html.Div([
                html.Label(col),
                dcc.Input(
                    id=col, 
                    type='number',
                    placeholder=f"Enter {col}",
                    value=0
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

            prediction = self.predict(data)

            if(prediction == 1):
                return True, 'The prediction is: Loan will not be repaid'
            else:
                return True, 'The prediction is: Loan will be repaid'

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
        port = int(os.environ.get("PORT", 10000))
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
        response = requests.post(self.PREDICT_URL, data=json_data, headers=headers)
        
        if response.status_code == 200:
            prediction = response.json()
            return int(prediction['prediction'])
        else:
            return None
