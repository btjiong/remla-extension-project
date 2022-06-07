"""
    Training the model

    This script:
        1) preprocesses the data
        2) generates the TF-IDF and/or BoW vectors
        3) trains the TF-IDF and/or BoW model
        4) evaluates the model
        5) saves the model
"""

from evaluation import get_evaluation_scores
from joblib import dump
from multilabel_classifier import train_classifier, transform_binary
from evaluation import get_evaluation_scores
from load_data import load_data, save_data
from data_validation import data_validation
from text_to_vector import bag_of_words, tfidf_features
from text_preprocessing import process_data

# 'data/' and 'output/' if running in docker
# '../data' and '../output/' if running this locally
data_dir = 'data/'
output_dir = 'output/'


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
    dump(tfidf_vectorizer, output_dir + 'tfidf_vectorizer.joblib')

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
    dump(classifier_tfidf, output_dir + 'tfidf_model.joblib')
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
    dump(classifier_mybag, output_dir + 'bow_model.joblib')
    print("============================================")


if __name__ == "__main__":
    # Loading and validating data
    print("Loading and validating data")
    train = data_validation(load_data(data_dir + 'train.tsv'))
    validation = data_validation(load_data(data_dir + 'validation.tsv'))
    test = data_validation(load_data(data_dir + 'test.tsv'))

    # Save validated data to file
    save_data(data_dir + 'train_new.tsv', train)
    save_data(data_dir + 'validation_new.tsv', validation)
    save_data(data_dir + 'test_new.tsv', test)

    # Preprocessing data
    print("Preprocessing the data...")
    x_train, y_train, x_val, y_val, x_test, tags_counts, words_counts = process_data(train, validation, test)

    # Transform labels to binary
    mlb, y_train, y_val = transform_binary(y_train, y_val, tags_counts)
    dump(mlb, output_dir + 'mlb.joblib')

    # Train and save the TF-IDF model
    train_tfidf(x_train, y_train, x_val, y_val, x_test)

    # Train and save the BoW model
    # train_bow(x_train, y_train, x_val, x_test, words_counts)

    print("Training is finished")
