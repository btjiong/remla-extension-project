"""
### Evaluation

To evaluate the results we will use several classification metrics:
 - [Accuracy](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
 - [F1-score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
 - [Area under ROC-curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
 - [Area under precision-recall curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)

Make sure you are familiar with all of them. How would you expect the things work for the multi-label scenario? Read about micro/macro/weighted averaging following the sklearn links provided above.
"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from text_preprocessing import get_data
from text_to_vector import bag_of_words, tfidf_features
from multilabel_classifier import train_classifier, transform_binary
from sklearn.metrics import roc_auc_score as roc_auc


def print_evaluation_scores(y_val, predicted):
    print('Accuracy score: ', accuracy_score(y_val, predicted))
    print('F1 score: ', f1_score(y_val, predicted, average='weighted'))
    print('Average precision score: ', average_precision_score(y_val, predicted, average='macro'))


if __name__ == '__main__':
    x_train, y_train, x_val, y_val, x_test, tags_counts, words_counts = get_data()

    X_train_mybag, X_val_mybag, X_test_mybag = bag_of_words(x_train, x_val, x_test, words_counts)

    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(x_train, x_val, x_test)
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

    #  Print: *accuracy*, *F1-score macro/micro/weighted*, *Precision macro/micro/weighted*
    print('Bag-of-words')
    print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
    print('Tfidf')
    print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)

    # Plot some generalization of the ROC curve for the case of multi-label classification
    roc_auc(y_val, y_val_predicted_scores_mybag, multi_class='ovo')
    roc_auc(y_val, y_val_predicted_scores_tfidf, multi_class='ovo')

    # coefficients = [0.1, 1, 10, 100]
    # penalties = ['l1', 'l2']
    #
    # for coefficient in coefficients:
    #     for penalty in penalties:
    #         classifier_tfidf = train_classifier(X_train_tfidf, y_train, penalty=penalty, C=coefficient)
    #         y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    #         y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
    #         print("Coefficient: {}, Penalty: {}".format(coefficient, penalty))
    #         print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)

    # Once we have the evaluation set up, we suggest that you experiment a bit with training your classifiers.
    # We will use *F1-score weighted* as an evaluation metric.
    classifier_tfidf = train_classifier(X_train_tfidf, y_train, penalty='l2', C=10)
    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

    test_predictions = classifier_tfidf.predict(X_test_tfidf)  ######### YOUR CODE HERE #############
    test_pred_inversed = mlb.inverse_transform(test_predictions)

    test_predictions_for_submission = '\n'.join(
        '%i\t%s' % (i, ','.join(row)) for i, row in enumerate(test_pred_inversed))