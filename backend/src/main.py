from flask import Flask, request, jsonify
from backend.src.data_processing.simple_load_data import SimpleLoadData
from backend.src.data_processing.simple_read_data import SimpleReadData
from backend.src.models.random_forest_loan_predictor import RandomForestLoanPredictor
import pandas as pd

app = Flask(__name__)
predictor = RandomForestLoanPredictor()
loader = SimpleLoadData()
reader = SimpleReadData()

FILES_FOLDER = 'data'

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Hello World!'}), 200

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    sampling_frequency = int(data['sampling_frequency'])
    target_variable = data['target_variable']
    concat = data['concat'] == 'True'

    loader.load(SimpleReadData.FILES_NAMES, FILES_FOLDER)
    loans = reader.read_data(FILES_FOLDER, concat, sampling_frequency)
    predictor.train(loans, target_variable)
    return jsonify({'message': 'Model trained successfully'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    loan = pd.DataFrame(data['loan'], index=[0])
    prediction = predictor.predict(loan)
    return jsonify({'prediction': prediction}), 200

@app.route('/evaluate', methods=['GET'])
def evaluate():
    accuracy = predictor.evaluate()
    return jsonify({'accuracy': accuracy}), 200

@app.route('/most_important_features', methods=['POST'])
def most_important_features():
    data = request.get_json()
    nb_features = data['nb_features']
    features = predictor.get_most_important_features(nb_features)
    return jsonify({'features': features.to_dict()}), 200

if __name__ == '__main__':
    app.run(debug=True)

