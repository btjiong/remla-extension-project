# Stack Overflow classifier

This is the back-end of the Stack Overflow classifier, used by the [Chrome extension](https://github.com/btjiong/remla-extension-frontend). The model is trained with the initial data, and used to make predictions on tags based on a received Stack Overflow title. The received title and the actual tags will be used to further train the model.

- [Instructions](#instructions)
  - [Running in Kubernetes cluster](#running-in-kubernetes-cluster)
  - [Running in Docker](#running-in-docker)
  - [Running locally](#running-locally)
- [Making a prediction](#making-a-prediction)
  - [Using the Chrome extension](#using-the-chrome-extension)
  - [Manually sending a request](#manually-sending-a-request)
- [Multilabel classification on Stack Overflow tags](#multilabel-classification-on-stack-overflow-tags)
  - [Dataset](#dataset)
  - [Transforming text to a vector](#transforming-text-to-a-vector)
  - [MultiLabel classifier](#multilabel-classifier)
  - [Evaluation](#evaluation)
  - [Libraries](#libraries)

# Instructions

## Running in Kubernetes cluster

1. Make sure that minikube is started and helm is installed
2. Run the shell script to deploy the pods

```
sh deploy-cluster.sh
```

3. If necessary use the minikube tunnel

```
minikube tunnel
```

4. The prediction endpoint should be at `localhost/predict`, and the API docs at `localhost/apidocs`. Prometheus should be at `localhost:9090`, and the Grafana dashboard should be at `localhost:3000`.

## Running in Docker

1. Build the docker image

```
docker build -t remla-extension-project .
```

2. Run the container

```
docker run -it --rm -p 5000:5000 remla-extension-project
```

3. The prediction endpoint should be at `localhost:5000/predict`, and the API docs at `localhost:5000/apidocs`.

## Running locally

1. Train the model by running train_model.py

```
python so_classifier/train_model.py
```

2. Start the flask API by running serve.py

```
python so_classifier/serve.py
```

3. The prediction endpoint should be at `localhost:5000/predict`, and the API docs at `localhost:5000/apidocs`.

# Making a prediction

## Using the Chrome extension

1. Download the [Chrome extension](https://github.com/btjiong/remla-extension-frontend).
2. Follow the instructions in the README.
3. Go to a Stack Overflow page to make a prediction.
4. The predicted tags and accuracy will be displayed below the title.

![image](https://user-images.githubusercontent.com/15816011/174408907-fab19f23-6e30-446a-9462-25f20db118c6.png)


## Manually sending a request

1. Send a POST request to the `/predict` endpoint locally, or to `https://so-classifier.herokuapp.com/predict`
2. The body should be a JSON in this format (tags are optional):

```json
{
    "title": "Ajax data - Uncaught ReferenceError date is not defined.",
    "tags": ["ajax", "javascript", "jquery"]
}
```

3. The response body is a JSON in this format:

```json
{
    "accuracy": 1.0,
    "result": ["ajax", "javascript", "jquery"],
    "title": "Ajax data - Uncaught ReferenceError date is not defined."
}
```

# Multilabel classification on Stack Overflow tags

Predict tags for posts from StackOverflow with multilabel classification approach.

## Dataset

-   Dataset of post titles from StackOverflow

## Transforming text to a vector

-   Transformed text data to numeric vectors using bag-of-words and TF-IDF.

## MultiLabel classifier

[MultiLabelBinarizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) to transform labels in a binary form and the prediction will be a mask of 0s and 1s.

[Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for Multilabel classification

-   Coefficient = 10
-   L2-regularization technique

## Evaluation

Results evaluated using several classification metrics:

-   [Accuracy](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
-   [F1-score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
-   [Area under ROC-curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
-   [Area under precision-recall curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)

## Libraries

-   [Numpy](http://www.numpy.org/) — a package for scientific computing.
-   [Pandas](https://pandas.pydata.org/) — a library providing high-performance, easy-to-use data structures and data analysis tools for the Python
-   [scikit-learn](http://scikit-learn.org/stable/index.html) — a tool for data mining and data analysis.
-   [NLTK](http://www.nltk.org/) — a platform to work with natural language.

Note: this sample project was originally created by @partoftheorigin
