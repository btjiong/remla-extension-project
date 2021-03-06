"""
    Flask API of the StackOverflow tag prediction model.

    predict (POST):
        in: "{"title": "title"}"
        out: "{"title": "title", "result": "tags"}"
"""
import joblib
from flasgger import Swagger
from flask import Flask, jsonify, make_response, request
from google.oauth2 import service_account
from googleapiclient.discovery import build

from so_classifier.model.text_preprocessing import text_prepare

app = Flask(__name__)
swagger = Swagger(app)

# Load model
tfidf_model = joblib.load("model/tfidf_model.joblib")
tfidf_vectorizer = joblib.load("model/tfidf_vectorizer.joblib")
mlb = joblib.load("model/mlb.joblib")

# Set up Google Drive API
SCOPES = ["https://www.googleapis.com/auth/drive"]
SERVICE_ACCOUNT_FILE = "so_classifier/credentials.json"
SPREADSHEET_ID = "1XeQkfdNCQB8L1EmwSEzgMeOSq3bXoKBh9JN337UGhSI"

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
service = build("sheets", "v4", credentials=credentials)

# Number of predictions and accuracy metrics
num_pred = 0

total_tp = 0
total_fn = 0
total_fp = 0
total_acc = 0


def add_pred():
    """
    Updates the number of predictions counter
    """
    global num_pred
    num_pred += 1


def get_pred():
    """
    Returns the number of predictions
    """
    return num_pred


def calculate_acc(pred, actual):
    """
    Calculates the individual accuracy of a single prediction
    """
    tp = 0
    fn = 0
    for x in actual:
        if x in pred:
            tp += 1
        else:
            fn += 1
    fp = len(pred) - tp
    return round(tp / (tp + fn + fp), 2)


def update_total_acc(acc):
    """
    Updates the average accuracy over all the predictions
    """
    global total_acc
    total_acc = round(((num_pred - 1) * total_acc + acc) / num_pred, 2)


def get_acc():
    """
    Returns the average accuracy
    """
    return total_acc


def upload_data(data):
    """
    Uploads the new title and tags to the online data set
    """
    service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range="A1:A2",
        body={"majorDimension": "ROWS", "values": data},
        valueInputOption="USER_ENTERED",
    ).execute()


@app.route("/predict", methods=["POST"])
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
            required: title
            properties:
                title:
                    type: string
                    example: Ajax data - Uncaught ReferenceError date is not defined.
    responses:
      200:
        description: "The result of the classification: tag(s)."
    """
    # Get the title and tags (if given) from the request
    input_data = request.get_json()
    title = input_data.get("title")
    tags = input_data.get("tags")

    # Apply preprocessing and vectorization to the title
    prepared_title = text_prepare(title)
    vectorized_title = tfidf_vectorizer.transform([prepared_title])

    # Make a prediction and retrieve the tags
    prediction = tfidf_model.predict(vectorized_title)
    prediction = mlb.inverse_transform(prediction)[0]

    add_pred()
    if tags is None:
        add_pred()
        return jsonify({"title": title, "result": prediction})

    # If tags are given, update the online data set
    upload_data(
        [
            [title, str(tags)],
        ]
    )

    # Calculate the accuracy and update the average accuracy
    accuracy = calculate_acc(prediction, tags)
    update_total_acc(accuracy)
    return jsonify({"title": title, "result": prediction, "accuracy": accuracy})


@app.route("/metrics", methods=["GET"])
def metrics():
    # Number of predictions
    text = "# HELP num_pred The total number of requested predictions.\n"
    text += "# TYPE num_pred counter\n"
    text += "num_pred " + str(get_pred()) + "\n\n"

    # Accuracy
    text += "# HELP acc Accuracy of the tag predictions.\n"
    text += "# TYPE acc gauge\n"
    text += "acc " + str(get_acc()) + "\n\n"

    response = make_response(text, 200)
    response.mimetype = "text/plain"
    return response


# Supressed a bandit B104 flag (open to non-local requests)
if __name__ == "__main__":
    app.run(host="0.0.0.0")  # nosec B104
