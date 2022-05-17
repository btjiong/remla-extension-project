"""
### Transforming text to a vector

Machine Learning algorithms work with numeric data and we cannot use the provided text data "as is".
There are many ways to transform text data to numeric vectors. In this task you will try to use two of them.
"""

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from text_preprocessing import get_data
from text_to_vector import bag_of_words, tfidf_features


def train_classifier(x_train, y_train, penalty='l1', C=1):
    """
      X_train, y_train — training data

      return: trained classifier
    """

    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    clf = LogisticRegression(penalty=penalty, C=C, dual=False, solver='liblinear')
    clf = OneVsRestClassifier(clf)
    clf.fit(x_train, y_train)

    return clf


def transform_binary(y_train, y_val, tags_counts):
    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)
    return mlb, y_train, y_val


if __name__ == '__main__':
    x_train, y_train, x_val, y_val, x_test, tags_counts, words_counts = get_data()

    X_train_mybag, X_val_mybag, X_test_mybag = bag_of_words(x_train, x_val, x_test, words_counts)

    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(x_train, x_val, x_test)
    tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}

    # Transform labels in a binary form, the prediction will be a mask of 0s and 1s.
    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)

    # Train the classifiers for different data transformations: *bag-of-words* and *tf-idf*.
    classifier_mybag = train_classifier(X_train_mybag, y_train)
    classifier_tfidf = train_classifier(X_train_tfidf, y_train)

    # Now you can create predictions for the data. You will need two types of predictions: labels and scores.
    y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
    y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

    # Now take a look at how classifier, which uses TF-IDF, works for a few examples:
    y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
    y_val_inversed = mlb.inverse_transform(y_val)
    for i in range(3):
        print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
            x_val[i],
            ','.join(y_val_inversed[i]),
            ','.join(y_val_pred_inversed[i])
        ))