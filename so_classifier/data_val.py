# put your import statements here
import pandas as pd


paths = ['../data/validation.tsv', '../data/train.tsv', '../data/test.tsv']


def read_data(p):
    df = pd.read_csv(p, sep='\t')
    read_data.cols = df.shape[1]
    print("Amount of columns detected: ", read_data.cols)
    if (int(read_data.cols)) == 2:
        titles = df['title']
        tags = df['tags']
        if (titles.isnull().sum() & tags.isnull().sum() == 0) and (len(titles) == len(tags)):
            pass
        else:
            print("Missing values are found, please check!")
        return titles, tags

    elif (int(read_data.cols)) == 1:
        titles = df['title']
        if (titles.isnull().sum()) == 0:
            pass
        else:
            print("Missing values are found, please check!")
        return titles


def only_check_title(p):
    titles = read_data(p)
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
    titles, tags = read_data(p)
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
    titles, tags = read_data(p)
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
    if int(read_data.cols) == 2:
        check_title.title_type = check_title(p)
        check_tags.tags_type = check_tags(p)

        new_titles = check_title.title_type
        new_tags = check_tags.tags_type
        new_df = pd.DataFrame({'title': new_titles, 'tags': new_tags})

        new_df.to_csv(p, index=False, sep='\t')
        print("saved checked .tsv file\n")
        return

    elif (int(read_data.cols)) == 1:
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
        read_data(paths[x])
        if (int(read_data.cols)) == 2:
            check_title(paths[x])
            check_tags(paths[x])
            save_as_tsv(paths[x])

        elif (int(read_data.cols)) == 1:
            only_check_title(paths[x])
            save_as_tsv(paths[x])

    print("--- CHECKING SUCCESSFULLY FINISHED ---")
    return


data_validation()




