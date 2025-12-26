from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
import json

DATA_FILE = "data.json"

models = {
    'KNN': './models/model_knn.pkl',
    'SVM': './models/model_svm.pkl',
    'DecisionTree': './models/model_decision_tree.pkl',
    'RandomForest': './models/model_random_forest.pkl',
    'XGBoost': './models/model_xgboost.pkl',
    'NaiveBayes': './models/model_naive_bayes.pkl',
    'MLP': './models/model_mlp.pkl',
}

def calculate_z_score(data, scaler):
    features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    feature_value = [data.get(feature, 0) for feature in features]

    z_score = scaler.transform([feature_value])[0]
    return dict(zip(features, z_score))

def load_file():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_record(record):
    records = load_file()
    records.append(record)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


app = Flask(__name__)

@app.route('/post/new/patient', methods=['POST'])
def post_patient():
    data = request.get_json()
    record = data.copy()

    model = joblib.load(models[data['model']])
    scaler = joblib.load('./models/zscore_scaler.joblib')


    if 'model' in data:
        data.pop('model')

    transformed_features = calculate_z_score(data, scaler)

    data.update(transformed_features)

    array_of_features = np.array(list(data.values())).reshape(1, -1)
    
    prediction = model.predict(array_of_features)

    record["result"] = int(prediction[0])

    save_record(record)

    del model, scaler
    return jsonify({"message": "Data recived", "result": prediction[0].item()})

@app.route('/get/patients', methods=['GET'])
def get_patients():
    data = load_file()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)

    
