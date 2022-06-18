"""
    Training the model

    This script:
        1) preprocesses the data
        2) generates the TF-IDF and/or BoW vectors
        3) trains the TF-IDF and/or BoW model
        4) evaluates the model
        5) saves the model
"""

from joblib import dump
from model.data_validation import data_validation
from model.evaluation import get_evaluation_scores
from model.load_data import concat_data, load_data, split_data
from model.multilabel_classifier import train_classifier, transform_binary
from model.text_preprocessing import process_data
from model.text_to_vector import bag_of_words, tfidf_features

# 'data/' and 'model/' if running in docker
# '../data' and '../model/' if running this locally
DATA_DIR = "data/"
OUTPUT_DIR = "model/"


def train_tfidf(x_train, y_train, x_val, y_val, x_test):
    """
    Trains the TF-IDF classifier, and saves the vectorizer and model
    """
    print("=============== TF-IDF model ===============")
    # TF-IDF features
    print("Generating TF-IDF features...")
    # pylint: disable=unused-variable
    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vectorizer = tfidf_features(
        x_train, x_val, x_test
    )
    dump(tfidf_vectorizer, OUTPUT_DIR + "tfidf_vectorizer.joblib")

    print("Training the TF-IDF classifier...")
    classifier_tfidf = train_classifier(X_train_tfidf, y_train)

    # Evaluating the models
    print("Evaluating the TF-IDF model:")
    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    accuracy, f1, avp = get_evaluation_scores(y_val, y_val_predicted_labels_tfidf)
    print("Accuracy: ", accuracy)
    print("F1-Score: ", f1)
    print("Average precision: ", avp)
    print("Saving the TF-IDF model...")
    dump(classifier_tfidf, OUTPUT_DIR + "tfidf_model.joblib")
    print("============================================")


def train_bow(x_train, y_train, x_val, x_test, words_counts):
    """
    Trains the BoW classifier, and saves the model
    """
    print("================ BoW model =================")
    # Bag of Words vectors
    print("Generating BoW vectors...")
    # pylint: disable=unused-variable
    X_train_mybag, X_val_mybag, X_test_mybag = bag_of_words(
        x_train, x_val, x_test, words_counts
    )

    # Train the classifiers for different data transformations: *bag-of-words* and *tf-idf*.
    print("Training the BoW classifier...")
    classifier_mybag = train_classifier(X_train_mybag, y_train)

    # Save the classifiers
    print("Saving the BoW model...")
    dump(classifier_mybag, OUTPUT_DIR + "bow_model.joblib")
    print("============================================")


if __name__ == "__main__":
    # Loading and validating data
    print("Loading and validating data")
    train = data_validation(load_data(DATA_DIR + "train.tsv"))
    validation = data_validation(load_data(DATA_DIR + "validation.tsv"))
    test = data_validation(load_data(DATA_DIR + "test.tsv"))
    online = data_validation(load_data(DATA_DIR + "online.tsv"))
    online_train, online_val, online_test = split_data(online)

    # Append the initial data with the new online data
    train = concat_data(train, online_train)
    validation = concat_data(validation, online_val)
    test = concat_data(test, online_test)

    # Save validated data to file
    # save_data(DATA_DIR + "train.tsv", train)
    # save_data(DATA_DIR + "validation.tsv", validation)
    # save_data(DATA_DIR + "test.tsv", test)

    # Preprocessing data
    print("Preprocessing the data...")
    x_train, y_train, x_val, y_val, x_test, tags_counts, words_counts = process_data(
        train, validation, test
    )

    # Transform labels to binary
    mlb, y_train, y_val = transform_binary(y_train, y_val, tags_counts)
    dump(mlb, OUTPUT_DIR + "mlb.joblib")

    # Train and save the TF-IDF model
    train_tfidf(x_train, y_train, x_val, y_val, x_test)

    # Train and save the BoW model
    # train_bow(x_train, y_train, x_val, x_test, words_counts)

    print("Training is finished")
