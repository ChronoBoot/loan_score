from user_interface_abc import UserInterface
from dash import Dash, html, Input, Output
from random_forest_loan_predictor import RandomForestLoanPredictor
import os
from dash import dcc

class DashUserInterface(UserInterface):
    def __init__(self, model: RandomForestLoanPredictor, categorical_values, float_values):
        self.app = Dash(__name__)
        self.model = model
        self.inputs_values = categorical_values 

        self.app.layout = html.Div([
            html.H1(
                children='Loan prediction',
                id='prediction',
                style={'textAlign': 'center'}
            ),
            *[dcc.Dropdown(
                id=col, 
                options=[{'label': val, 'value': val} for val in values],
                placeholder=f"Select {col}"
            ) for col, values in self.inputs_values.items()],
            *[dcc.Input(
                id=col, 
                type='number',
                placeholder=f"Enter {col}"
            ) for col in float_values],
            html.Button('Predict', id='predict-button', n_clicks=0),
        ])

        @self.app.callback(
            Output('prediction', 'children'),
            Input('predict-button', 'n_clicks')
        )
        def update(n_clicks):
            if n_clicks > 0:
                return self.predict()

    def display(self):
        port = int(os.environ.get("PORT", 10000))
        host = os.getenv("HOST", '0.0.0.0')
        self.app.run_server(debug=True, host=host, port=port)

    def predict(self):
        return self.model.predict([])

   
        