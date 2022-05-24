"""
Flask API of the SMS Spam detection model model.
"""
import traceback
import joblib
from flask import Flask, jsonify, request
from flasgger import Swagger
import pandas as pd


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from text_preprocessing import get_data
from text_to_vector import bag_of_words, tfidf_features
from multilabel_classifier import train_classifier

app = Flask(__name__)
swagger = Swagger(app)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict whether an SMS is Spam.
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
                sms:
                    type: string
                    example: This is an example of an SMS.
    responses:
      200:
        description: "The result of the classification: 'spam' or 'ham'."
    """
    # x_train, y_train, x_val, y_val, x_test, tags_counts, words_counts = get_data()
    #
    # X_train_mybag, X_val_mybag, X_test_mybag = bag_of_words(x_train, x_val, x_test, words_counts)
    #
    # X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(x_train, x_val, x_test)
    # tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}
    #
    # # Transform labels in a binary form, the prediction will be a mask of 0s and 1s.
    # mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    # y_train = mlb.fit_transform(y_train)
    # y_val = mlb.fit_transform(y_val)
    #
    # # Train the classifiers for different data transformations: *bag-of-words* and *tf-idf*.
    # classifier_mybag = train_classifier(X_train_mybag, y_train)
    # classifier_tfidf = train_classifier(X_train_tfidf, y_train)
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


if __name__ == '__main__':
    # clf = joblib.load('output/model.joblib')

    app.run(port=8080, debug=True)