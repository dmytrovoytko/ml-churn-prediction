from flask import Flask, jsonify, redirect, request, url_for

from settings import DATA_DIR, MODEL_DIR, TARGET, PORT  # isort:skip

from settings import DEBUG  # isort:skip

DEBUG = True  # True # False # override global settings

from predict import predict_dict
from preprocess import enc_load, preprocess_data

app = Flask('online-prediction')


def calculate_prediction(object):
    try:
        result = predict_dict(object, verbose=True)
        return jsonify(result)
    except Exception as e:
        print('!! predict_dict Error:', e)
        return "Wrong data/dataset not recognized", 400  # status.HTTP_400_BAD_REQUEST


@app.route('/predict', methods=['POST'])
def predict():
    object = request.get_json()
    if DEBUG:
        print('Received request:', object)
    return calculate_prediction(object)


@app.route('/', methods=['GET'])
def root():
    return {'Prediction':'works!'}


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=PORT)
