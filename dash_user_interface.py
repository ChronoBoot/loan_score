from user_interface_abc import UserInterface
from dash import Dash, html, Input, Output, State, dcc, callback, no_update
from random_forest_loan_predictor import RandomForestLoanPredictor
import os
import pandas as pd

class DashUserInterface(UserInterface):
    def __init__(self, model: RandomForestLoanPredictor, categorical_values, float_values):
        self.app = Dash(__name__)
        self.model = model
        self.categorical_values = categorical_values
        self.float_values = float_values 

        self.app.layout = html.Div([
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
            ]) for col in self.float_values],
            html.Button('Predict', id='predict-button', n_clicks=0),
            dcc.ConfirmDialog(
                id='prediction-popup',
                message='',
                displayed=False
            )
        ])

        @self.app.callback(
            Output('prediction-popup', 'displayed'),
            Output('prediction-popup', 'message'),
            Input('predict-button', 'n_clicks'),
            [State(dropdown, 'value') for dropdown in self.categorical_values.keys()],
            [State(input, 'value') for input in self.float_values]
        )
        def update(n_clicks, *args):
            if n_clicks > 0:
                nb_categorical_values = len(self.categorical_values.keys())

                categorical_values = [val for val in args[:nb_categorical_values]]
                float_values = [val for val in args[nb_categorical_values:]]

                data = pd.DataFrame([categorical_values + float_values], columns=list(self.categorical_values.keys()) + self.float_values)
                
                prediction = self.predict(data)
                return True, f'The prediction is: {prediction}'

            return no_update, ''

        @self.app.callback(
            Output('prediction', 'children'),
            Input('predict-button', 'n_clicks')
        )
        def clear_prediction(n_clicks):
            if n_clicks > 0:
                return ''
            return no_update

    def display(self):
        port = int(os.environ.get("PORT", 10000))
        host = os.getenv("HOST", '0.0.0.0')
        self.app.run_server(debug=True, host=host, port=port)

    def predict(self, data=[]):
        return self.model.predict(data)
        def update(n_clicks, *args):
            if n_clicks > 0:
                nb_categorical_values = len(self.categorical_values.keys())

                categorical_values = [val for val in args[:nb_categorical_values]]
                float_values = [val for val in args[nb_categorical_values:]]

                data = pd.DataFrame([categorical_values + float_values], columns=list(self.categorical_values.keys()) + self.float_values)
                
                return self.predict(data)

    def display(self):
        port = int(os.environ.get("PORT", 10000))
        host = os.getenv("HOST", '0.0.0.0')
        self.app.run_server(debug=True, host=host, port=port)

    def predict(self, data=[]):
        return self.model.predict(data)
