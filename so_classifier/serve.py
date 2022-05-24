"""
Flask API of the StackOverflow tag prediction model.
"""
from flask import Flask, jsonify, request
from flasgger import Swagger

import evaluation

app = Flask(__name__)
swagger = Swagger(app)


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
                    example: This is an example of an StackOverflow title.
    responses:
      200:
        description: "The result of the classification: tag."
    """
    input_data = request.get_json()
    title = input_data.get('title')
    # prediction = classifier_tfidf.predict(title)
    # processed_title = prepare(title)
    # model = joblib.load('output/model.joblib')
    # prediction = model.predict(processed_title)[0]
    #
    # return jsonify({
    #     "result": prediction,
    #     "title": title
    # })
    return jsonify({
        "title": title,
        "test": "OK"
    })

@app.route('/evaluate', methods=['GET'])
def evaluate():
    """
    Evaluate the model
    ---
    consumes:
      - application/json
    responses:
      200:
        description: "The evaluation of the BoW and TF-IDF classification."
    """

    tfidfAcc, tfidfF1, tfidfAvp = evaluation.evaluate()

    return jsonify({
        # "BoW Accuracy": bowAcc,
        # "BoW F1-Score": bowF1,
        # "BoW Average Precision": bowAvp,
        "TF-IDF Accuracy": tfidfAcc,
        "TF-IDF F1-Score": tfidfF1,
        "TF-IDF Average Precision": tfidfAvp
    })


if __name__ == '__main__':
    # clf = joblib.load('output/model.joblib')

    app.run(port=8080, debug=True)