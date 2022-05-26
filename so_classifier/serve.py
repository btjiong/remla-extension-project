"""
Flask API of the StackOverflow tag prediction model.
"""
import joblib
from flask import Flask, jsonify, request
from flasgger import Swagger

from text_preprocessing import text_prepare

app = Flask(__name__)
swagger = Swagger(app)

tfidf_model = joblib.load('output/tfidf_model.joblib')
tfidf_vectorizer = joblib.load('output/tfidf_vectorizer.joblib')
mlb = joblib.load('output/mlb.joblib')


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

    return jsonify({
        "title": title,
        "result": prediction
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0')
