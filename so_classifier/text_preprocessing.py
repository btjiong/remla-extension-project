"""
    Text preprocessing

    Functions for reading and preprocessing the data. Use these functions to get the resulting datasets, and the
    tags and words counts.
"""

import nltk
from nltk.corpus import stopwords
from ast import literal_eval
import pandas as pd
import re

# Download stopwords
nltk.download('stopwords')
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def read_data(filename):
    """
        filename: the filename

        return: the data from the file
    """
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data


def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, "", text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join([word for word in text.split() if not word in STOPWORDS])  # delete stopwords from text
    return text


def count_words(corpus):
    """
        corpus: the input data

        return: the words counts
    """
    words_counts = {}
    for sentence in corpus:
        for word in sentence.split():
            if word in words_counts:
                words_counts[word] += 1
            else:
                words_counts[word] = 1
    return words_counts


def count_tags(corpus):
    """
        corpus: the input data

        return: the tags counts
    """
    tags_counts = {}
    for tags in corpus:
        for tag in tags:
            if tag in tags_counts:
                tags_counts[tag] += 1
            else:
                tags_counts[tag] = 1
    return tags_counts


def get_data():
    """
        return: the preprocessed data, and the tags and words counts
    """
    # You are provided a split to 3 sets: *train*, *validation* and *test*.
    # All corpora (except for *test*) contain titles of the posts and corresponding tags (100 tags are available).
    train = read_data('data/train.tsv')
    validation = read_data('data/validation.tsv')
    test = pd.read_csv('data/test.tsv', sep='\t')

    # For a more comfortable usage, initialize *X_train*, *X_val*, *X_test*, *y_train*, *y_val*.
    x_train, y_train = train['title'].values, train['tags'].values
    x_val, y_val = validation['title'].values, validation['tags'].values
    x_test = test['title'].values

    # Now we can preprocess the titles using function *text_prepare* and
    # making sure that the headers don't have bad symbols:
    x_train = [text_prepare(x) for x in x_train]
    x_val = [text_prepare(x) for x in x_val]
    x_test = [text_prepare(x) for x in x_test]

    tags_counts = count_tags(y_train)
    words_counts = count_words(x_train)

    return x_train, y_train, x_val, y_val, x_test, tags_counts, words_counts
