from joblib import dump, load

from multilabel_classifier import train_classifier, transform_binary
from text_to_vector import bag_of_words, tfidf_features
from text_preprocessing import get_data

if __name__ == "__main__":
    # Preprocessing data
    print("Preprocessing the data...")
    x_train, y_train, x_val, y_val, x_test, tags_counts, words_counts = get_data()

    # Bag of Words vectors
    print("Generating BoW vectors...")
    X_train_mybag, X_val_mybag, X_test_mybag = bag_of_words(x_train, x_val, x_test, words_counts)

    # TF-IDF features
    print("Generating TF-IDF features...")
    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(x_train, x_val, x_test)

    # Transform labels in a binary form
    mlb, y_train, y_val = transform_binary(y_train, y_val, tags_counts)

    # Train the classifiers for different data transformations: *bag-of-words* and *tf-idf*.
    print("Training the BoW classifier...")
    classifier_mybag = train_classifier(X_train_mybag, y_train)
    print("Training the TF-IDF classifier...")
    classifier_tfidf = train_classifier(X_train_tfidf, y_train)

    # Save the classifiers
    print("Saving the BoW model...")
    dump(classifier_mybag, '../output/bow_model.joblib')
    print("Saving the TF-IDF model...")
    dump(classifier_tfidf, '../output/tfidf_model.joblib')