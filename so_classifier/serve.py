"""
    Flask API of the StackOverflow tag prediction model.

    predict (POST):
        in: "{"title": "title"}"
        out: "{"title": "title", "result": "tags"}"
"""
import joblib
from flask import Flask, jsonify, request
from flasgger import Swagger

from text_preprocessing import text_prepare

app = Flask(__name__)
swagger = Swagger(app)

tfidf_model = joblib.load('../output/tfidf_model.joblib')
tfidf_vectorizer = joblib.load('../output/tfidf_vectorizer.joblib')
mlb = joblib.load('../output/mlb.joblib')

total_tp = 0
total_fn = 0
total_fp = 0
total_accuracy = 0

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict whether the tag of a StackOverflow title.
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: sms
            properties:
                title:
                    type: string
                    example: Ajax data - Uncaught ReferenceError date is not defined.
    responses:
      200:
        description: "The result of the classification: tag(s)."
    """
    input_data = request.get_json()
    title = input_data.get('title')
    tags = input_data.get('tags')
    prepared_title = text_prepare(title)
    vectorized_title = tfidf_vectorizer.transform([prepared_title])
    prediction = tfidf_model.predict(vectorized_title)
    prediction = mlb.inverse_transform(prediction)[0]

    if tags is None:
        return jsonify({
            "title": title,
            "result": prediction
        })

    accuracy = calculate_acc(prediction, tags)
    global total_accuracy
    return jsonify({
        "title": title,
        "result": prediction,
        "accuracy": accuracy,
        "total_accuracy": total_accuracy
    })


def calculate_acc(pred, actual):
    tp = 0
    fn = 0
    for x in actual:
        if x in pred:
            tp += 1
        else:
            fn += 1
    fp = len(pred) - tp
    update_acc(tp, fn, fp)
    return round(tp / (tp + fn + fp), 2)


def update_acc(tp, fn, fp):
    global total_tp
    global total_fn
    global total_fp
    global total_accuracy
    total_tp += tp
    total_fn += fn
    total_fp += fp
    total_accuracy = round(total_tp / (total_tp + total_fn + total_fp), 2)


if __name__ == '__main__':
    # app.run(host='0.0.0.0')
    app.run(port='5000')