# put your import statements here
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data():
    #hier zorgen dat de functie alle 3 path x 2 columns exporteert
    # train_path = '../data/validation.tsv'
    # validation_path = '../data/validation.tsv'
    # test_path = '../data/validation.tsv'
    # paths = [train_path, validation_path, test_path]
    # for path in range(len(paths)):

    df = pd.read_csv('../data/validation.tsv', sep='\t')
    titles = df['title']
    tags = df['tags']
    return titles, tags


def check_title():
    titles, tags = read_data()
    print('title check')
    check_tags.tags_type = tags
    check_title.title_type = titles
    for i in range(1, len(check_title.title_type)): #begins at 1 since 0 are column headers title and tags
        if (type(check_title.title_type[i])) == str:
            pass
        else:
            print(i, check_title.title_type[i], 'has is no string,and is now deleted')
            check_title.title_type.delete(i)
            check_tags.tags_type.delete(i)
    return check_title.title_type


def check_tags():
    titles, tags = read_data()
    check_tags.tags_type = tags
    check_title.title_type = titles
    print('tag check')
    check_tags.tags_type = tags
    for i in range(1, len(check_tags.tags_type)): #begins at 1 since 0 are column headers title and tags
        if (type(check_tags.tags_type[i])) == str: #vgm worden deze tags gezien als strings en niet als list:( !!
            sep_list = ["[", "]", "'"]

            for sep in range(len(sep_list)):
                if check_tags.tags_type[i].find(sep_list[sep]) != -1:
                    pass
                else:
                    print("tag seperator not found in: ", check_tags.tags_type[i])
                    check_tags.tags_type.delete(i)
                    check_title.title_type.delete(i)
        else:
            print(i, check_tags.tags_type[i], 'tag is no str,and is now deleted')
            check_tags.tags_type.delete(i)
            check_title.title_type.delete(i)
    return check_tags.tags_type


def save_as_tsv():
    check_title.title_type = check_title()
    check_tags.tags_type = check_tags()

    new_titles = check_title.title_type
    new_tags = check_tags.tags_type
    new_df = new_titles + "," + new_tags
    print("new csv file")
    print(new_df)
    new_df.to_csv('../data/train.tsv', sep='\t')
    return


# def data_validation():
#     for tsv in range(len(paths)):
#         check_title(tsv)
#         check_tags(tsv)
#         save_as_tsv(path)
#
#     return


check_title()
check_tags()
save_as_tsv()

#--------TO DOs-----------
#zorgen dat als bij de "title column" een item wordt verwijderd, hetzelfde row item ook bij "tags column" wordt verwijderd
#en vise versa
#miss in de function een variabele maken voor de test, train en validation file zodat al die bestand te checken zijn





