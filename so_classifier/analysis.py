"""
    Analysis of the most important features
"""

from multilabel_classifier import train_classifier, transform_binary
from text_preprocessing import get_data
from text_to_vector import bag_of_words, tfidf_features


def print_words_for_tag(classifier, tag, tags_classes, index_to_words):
    """
    classifier: trained classifier
    tag: particular tag
    tags_classes: a list of classes names from MultiLabelBinarizer
    index_to_words: index_to_words transformation
    all_words: all words in the dictionary

    return nothing, just print top 5 positive and top 5 negative words for current tag
    """
    print("Tag:\t{}".format(tag))

    # Extract an estimator from the classifier for the given tag.
    # Extract feature coefficients from the estimator.

    model = classifier.estimators_[tags_classes.index(tag)]
    top_positive_words = [
        index_to_words[x] for x in model.coef_.argsort().tolist()[0][-5:]
    ]
    top_negative_words = [
        index_to_words[x] for x in model.coef_.argsort().tolist()[0][:5]
    ]

    print("Top positive words:\t{}".format(", ".join(top_positive_words)))
    print("Top negative words:\t{}\n".format(", ".join(top_negative_words)))


if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, tags_counts, words_counts = get_data()

    X_train_mybag, X_val_mybag, X_test_mybag = bag_of_words(
        x_train, x_val, x_test, words_counts
    )

    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(
        x_train, x_val, x_test
    )
    tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}

    # Transform labels in a binary form, the prediction will be a mask of 0s and 1s.
    mlb, y_train, y_val = transform_binary(y_train, y_val, tags_counts)

    # Train the classifiers for different data transformations: *bag-of-words* and *tf-idf*.
    classifier_mybag = train_classifier(X_train_mybag, y_train)
    classifier_tfidf = train_classifier(X_train_tfidf, y_train)

    # Now you can create predictions for the data. You will need two types of predictions: labels and scores.
    y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
    y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

    classifier_tfidf = train_classifier(X_train_tfidf, y_train, penalty="l2", C=10)

    print_words_for_tag(classifier_tfidf, "c", mlb.classes, tfidf_reversed_vocab)
    print_words_for_tag(classifier_tfidf, "c++", mlb.classes, tfidf_reversed_vocab)
    print_words_for_tag(classifier_tfidf, "linux", mlb.classes, tfidf_reversed_vocab)
