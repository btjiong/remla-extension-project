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

# --- NEWLY ADDED DATA VALIDATION CODE ---


paths = ['../data/validation.tsv', '../data/train.tsv', '../data/test.tsv']


def load_data(p):
    df = pd.read_csv(p, sep='\t')
    load_data.cols = df.shape[1]
    print("Amount of columns detected: ", load_data.cols)
    if (int(load_data.cols)) == 2:
        titles = df['title']
        tags = df['tags']
        if (titles.isnull().sum() & tags.isnull().sum() == 0) and (len(titles) == len(tags)):
            pass
        else:
            print("Missing values are found, please check!")
        return titles, tags

    elif (int(load_data.cols)) == 1:
        titles = df['title']
        if (titles.isnull().sum()) == 0:
            pass
        else:
            print("Missing values are found, please check!")
        return titles


def only_check_title(p):
    titles = load_data(p)
    print('Checking the test titles')
    only_check_title.title_type = titles
    for i in range(len(only_check_title.title_type)):
        if (type(only_check_title.title_type[i])) == str:
            pass
        else:
            print(i, only_check_title.title_type[i], 'has is no string and is now deleted')
            only_check_title.title_type.delete(i)
    print("checking done")
    return only_check_title.title_type


def check_title(p):
    titles, tags = load_data(p)
    print('checking titles')
    check_tags.tags_type = tags
    check_title.title_type = titles
    for i in range(len(check_title.title_type)):
        if (type(check_title.title_type[i])) == str:
            pass
        else:
            print(i, check_title.title_type[i], 'has is no string and is now deleted')
            check_title.title_type.delete(i)
            check_tags.tags_type.delete(i)
    print("Checking titles done")
    return check_title.title_type


def check_tags(p):
    titles, tags = load_data(p)
    check_tags.tags_type = tags
    check_title.title_type = titles

    print('Checking tags')
    for i in range(len(check_tags.tags_type)):
        if (type(check_tags.tags_type[i])) == str:
            sep_list = ["[", "]", "'"]

            for sep in range(len(sep_list)):
                if check_tags.tags_type[i].find(sep_list[sep]) != -1:
                    pass
                else:
                    print("tag seperator not found in: ", check_tags.tags_type[i])
                    check_tags.tags_type.delete(i)
                    check_title.title_type.delete(i)
        else:
            print(i, check_tags.tags_type[i], 'tag is no string and is now deleted')
            check_tags.tags_type.delete(i)
            check_title.title_type.delete(i)
    print("checking tags done")
    return check_tags.tags_type


def save_as_tsv(p):
    print("Now saving")
    if int(load_data.cols) == 2:
        check_title.title_type = check_title(p)
        check_tags.tags_type = check_tags(p)

        new_titles = check_title.title_type
        new_tags = check_tags.tags_type
        new_df = pd.DataFrame({'title': new_titles, 'tags': new_tags})

        new_df.to_csv(p, index=False, sep='\t')
        print("saved checked .tsv file\n")
        return

    elif (int(load_data.cols)) == 1:
        only_check_title.title_type = only_check_title(p)
        new_titles = only_check_title.title_type

        new_df = pd.DataFrame({'title': new_titles})
        new_df.to_csv(p, index=False, sep='\t')
        print("saved checked .tsv test file\n")
        return


def data_validation():
    print("--- DATA VALIDATION STARTED ---")
    for x in range(len(paths)):
        print("NOW CHECKING PATH #", x + 1, paths[x])
        load_data(paths[x])
        if (int(load_data.cols)) == 2:
            check_title(paths[x])
            check_tags(paths[x])
            save_as_tsv(paths[x])

        elif (int(load_data.cols)) == 1:
            only_check_title(paths[x])
            save_as_tsv(paths[x])

    print("--- CHECKING SUCCESSFULLY FINISHED ---")
    return


data_validation()
