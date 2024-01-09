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
DATA_FILE_MODEL = 'data_for_model.csv'
DATA_FILE_TESTING = 'data_for_testing.csv'
COMMON_STRUCTURE_PATH = 'shared_config'
JSON_FILE_STRUCTURE = 'data_structure.json'

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Hello World!'}), 200

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    sampling_frequency = int(data['sampling_frequency'])
    target_variable = data['target_variable']
    
    loader.load(SimpleReadData.FILES_NAMES, FILES_FOLDER)
    reader.write_data(FILES_FOLDER, DATA_FILE_MODEL, sampling_frequency)

    loans = reader.read_data(FILES_FOLDER, DATA_FILE_MODEL)
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

@app.route('/write_model_data', methods=['GET'])
def write_model_data(frequency: int):
    reader.write_data(FILES_FOLDER, DATA_FILE_MODEL, frequency)
    return jsonify({'message': 'Model data written successfully'}), 200

@app.route('/generate_structure', methods=['GET'])
def generate_structure():
    data = reader.read_data(FILES_FOLDER, DATA_FILE_MODEL)
    reader.write_data_structure_json(data, COMMON_STRUCTURE_PATH, JSON_FILE_STRUCTURE)
    return jsonify({'message': 'Structure generated successfully'}), 200

@app.route('/get_loan_example', methods=['GET'])
def get_loan_example():
    reader.write_data(FILES_FOLDER, DATA_FILE_TESTING, 10000, False)
    loan_example = reader.read_data(FILES_FOLDER, DATA_FILE_TESTING)
    loan_json = loan_example.to_json(orient='records')
    return jsonify({'loan_example': loan_json}), 200

if __name__ == '__main__':
    app.run(debug=True)
