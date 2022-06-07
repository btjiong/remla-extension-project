from ast import literal_eval

import pandas as pd
from load_data import load_data

paths = ['../data/validation.tsv', '../data/train.tsv', '../data/test.tsv']


# loading the dataframe from load_data.py and checking the amount of columns
def column_check(data, cols):
    print("Amount of columns detected: ", cols)
    if (int(cols)) == 2:
        titles = data['title']
        tags = data['tags']
        if (titles.isnull().sum() + tags.isnull().sum() == 0) and (len(titles) == len(tags)):
            pass
        else:
            print("Missing values are found, please check!")
        return titles, tags

    elif (int(cols)) == 1:
        titles = data['title']
        if (titles.isnull().sum()) == 0:
            pass
        else:
            print("Missing values are found, please check!")
        return titles


# checking the titles
def check_title(titles, tags=None):
    print('Checking titles')
    for i in range(len(titles)):
        curr_title = titles[i]
        if isinstance(curr_title, str):
            pass
        else:
            print(i, curr_title, 'has is no string and is now deleted')
            titles.delete(i)
            if tags is not None:
                tags.delete(i)

    print("Checking titles done")
    return titles, tags


# checking the tags
def check_tags(titles, tags):
    print('Checking tags')
    for i in range(len(tags)):
        curr_tag = tags[i]
        if isinstance(curr_tag, str):
            sep_list = ["[", "]", "'"]

            for sep in range(len(sep_list)):
                if curr_tag.find(sep_list[sep]) != -1:
                    pass
                else:
                    print("tag seperator not found in: ", curr_tag)
                    tags.delete(i)
                    titles.delete(i)
        else:
            print(i, curr_tag, 'tag is no string and is now deleted')
            tags.delete(i)
            titles.delete(i)
    print("Checking tags done")
    return titles, tags


# running the functions in order to validate the data
def data_validation(data):
    print("--- DATA VALIDATION STARTED ---")
    cols = int(data.shape[1])
    if int(cols) == 2:
        titles, tags = column_check(data, cols)
        titles, tags = check_title(titles, tags)
        titles, tags = check_tags(titles, tags)
        df = pd.DataFrame({'title': titles, 'tags': tags})
        df['tags'] = df['tags'].apply(literal_eval)
        return df

    elif int(cols) == 1:
        titles = column_check(data, cols)
        titles, tags = check_title(titles)
        df = pd.DataFrame({'title': titles})
        return df

    else:
        print("Incorrect amount of columns")
        # TODO THROW EXCEPTION HERE


# data_validation(load_data(paths[0]))
