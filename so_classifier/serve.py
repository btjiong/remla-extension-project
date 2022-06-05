"""
    Flask API of the StackOverflow tag prediction model.

    predict (POST):
        in: "{"title": "title"}"
        out: "{"title": "title", "result": "tags"}"
"""
import joblib
from flask import Flask, jsonify, request, make_response
from flasgger import Swagger

from so_classifier.text_preprocessing import text_prepare

app = Flask(__name__)
swagger = Swagger(app)

tfidf_model = joblib.load('output/tfidf_model.joblib')
tfidf_vectorizer = joblib.load('output/tfidf_vectorizer.joblib')
mlb = joblib.load('output/mlb.joblib')

numPred = 0


def addPred():
    global numPred
    numPred += 1


def getPred():
    global numPred
    return numPred


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
    prepared_title = text_prepare(title)
    vectorized_title = tfidf_vectorizer.transform([prepared_title])
    prediction = tfidf_model.predict(vectorized_title)
    prediction = mlb.inverse_transform(prediction)
    addPred()

    return jsonify({
        "title": title,
        "result": prediction
    })


@app.route('/metrics', methods=['GET'])
def metrics():
    text = "# HELP num_pred The total number of requested predictions.\n"
    text += "# TYPE num_pred counter\n"
    text += "num_pred " + str(getPred()) + "\n\n"
    response = make_response(text, 200)
    response.mimetype = "text/plain"
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0')
