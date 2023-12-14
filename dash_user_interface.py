from user_interface_abc import UserInterface
from dash import Dash, html
import os

class DashUserInterface(UserInterface):
    def __init__(self):
        self.app = Dash(__name__)
        self.app.layout = html.Div([html.H1(children='Loan prediction', style={'textAlign':'center'})])

    def display(self):
        port = int(os.environ.get("PORT", 10000))
        host = os.getenv("HOST", '0.0.0.0')
        self.app.run_server(debug=True, host=host, port=port)

    def send_data(self):
        raise NotImplementedError("The send_data method is not implemented")