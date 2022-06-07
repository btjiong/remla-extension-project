"""
    Training the classifiers

    Functions for training the classifiers, and transforming features into binary.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


def train_classifier(x_train, y_train, penalty="l1", C=1):
    """
    Create and fit LogisticRegression wrapped into OneVsRestClassifier.

    X_train, y_train: training data
    penalty: the penalty
    C: C

    return: trained classifier
    """
    clf = LogisticRegression(penalty=penalty, C=C, dual=False, solver="liblinear")
    clf = OneVsRestClassifier(clf)
    clf.fit(x_train, y_train)

    return clf


def transform_binary(y_train, y_val, tags_counts):
    """
    y_train, y_val: the labels
    tags_counts: the tags counts

    return: the multilabel binarizer, and the transformed label sets
    """
    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)
    return mlb, y_train, y_val
