"""
    Transforming text to a vector

    Functions for fitting the BoW and TF-IDF vectorizers, and using these fitted vectorizers for transforming
    StackOverflow titles into vectors/features.

"""
import joblib
import numpy as np
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def my_bag_of_words(text, words_to_index, dict_size):
    """
    text: a string
    words_to_index: the dictionary
    dict_size: size of the dictionary

    return: a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)

    for word in text.split():
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1
    return result_vector


def bow_dict(size, words_counts):
    """
    Generate a dictionary based on the words counts

    size: the size of the dictionary
    words_counts: the words counts

    return: the dictionary size, index to words, and words to index
    """
    dic_size = size
    index_to_words = sorted(words_counts, key=words_counts.get, reverse=True)[
        :dic_size
    ]  ####### YOUR CODE HERE #######
    words_to_index = {word: i for i, word in enumerate(index_to_words)}
    return dic_size, index_to_words, words_to_index


def bag_of_words(x_train, x_val, x_test, words_counts):
    """
    Get BoW vectors for training, validation and test set

    x_train, x_val, x_test: samples
    words_counts: the words counts

    return: BoW vectorized representation of each sample
    """
    dic_size, index_to_words, words_to_index = bow_dict(5000, words_counts)

    x_train_mybag = sp_sparse.vstack(
        [
            sp_sparse.csr_matrix(my_bag_of_words(text, words_to_index, dic_size))
            for text in x_train
        ]
    )
    x_val_mybag = sp_sparse.vstack(
        [
            sp_sparse.csr_matrix(my_bag_of_words(text, words_to_index, dic_size))
            for text in x_val
        ]
    )
    x_test_mybag = sp_sparse.vstack(
        [
            sp_sparse.csr_matrix(my_bag_of_words(text, words_to_index, dic_size))
            for text in x_test
        ]
    )

    return x_train_mybag, x_val_mybag, x_test_mybag


def bow_transform(dataset, words_counts):
    """
    Transform a set of StackOverflow titles into BoW vectors

    dataset: samples
    words_counts: words counts

    return: BoW vectorized representation of each sample
    """
    dic_size, index_to_words, words_to_index = bow_dict(5000, words_counts)

    mybag = sp_sparse.vstack(
        [
            sp_sparse.csr_matrix(my_bag_of_words(text, words_to_index, dic_size))
            for text in dataset
        ]
    )

    return mybag


def tfidf_features(x_train, x_val, x_test):
    """
    Create TF-IDF vectorizer with a proper parameters choice
    Fit the vectorizer on the train set
    Transform the train, test, and val sets and return the result

    x_train, x_val, x_test: samples

    return: TF-IDF vectorized representation of each sample and vocabulary
    """

    tfidf_vectorizer = TfidfVectorizer(
        min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern="(\S+)"
    )  ####### YOUR CODE HERE #######

    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
    x_val_tfidf = tfidf_vectorizer.transform(x_val)
    x_test_tfidf = tfidf_vectorizer.transform(x_test)

    return x_train_tfidf, x_val_tfidf, x_test_tfidf, tfidf_vectorizer


def tfidf_transform(dataset):
    """
    Transform a set of StackOverflow titles into TF-IDF features

    dataset: samples

    return: TF-IDF vectorized representation of each sample and vocabulary
    """
    tfidf_vectorizer = joblib.load("../output/tfidf_vectorizer.joblib")

    dataset_tfidf = tfidf_vectorizer.transform(dataset)

    return dataset_tfidf
