"""
    Evaluation

    To evaluate the results we will use several classification metrics:
     - [Accuracy](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
     - [F1-score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
     - [Area under ROC-curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
     - [Area under precision-recall curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)
"""

from sklearn.metrics import accuracy_score, average_precision_score, f1_score


def get_evaluation_scores(y_val, predicted):
    """
    y_val: the actual labels
    predicted: the predicted labels

    return: the accuracy, f1-score, and average precision
    """
    accuracy = accuracy_score(y_val, predicted)
    f1 = f1_score(y_val, predicted, average="weighted")
    avp = average_precision_score(y_val, predicted, average="macro")
    return accuracy, f1, avp
