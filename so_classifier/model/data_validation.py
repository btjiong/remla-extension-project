from ast import literal_eval


def column_check(data, cols):
    """
    Loads the DataFrame and checks the amount of columns
    """
    print("Amount of columns detected: ", cols)
    if (int(cols)) == 2:
        titles = data["title"]
        tags = data["tags"]
        if (titles.isnull().sum() + tags.isnull().sum() != 0) or (
            len(titles) != len(tags)
        ):
            print("Missing values are found, please check!")
            # TODO Throw exception

    elif (int(cols)) == 1:
        titles = data["title"]
        if (titles.isnull().sum()) != 0:
            print("Missing values are found, please check!")
            # TODO Throw exception


def check_title(data):
    """
    Checks if the titles in the DataFrame are in the correct format
    """
    print("Checking titles")
    titles = data["title"].values
    for i in range(len(titles)):
        curr_title = titles[i]
        if isinstance(curr_title, str):
            if len(curr_title) > 150:
                print(
                    f"The title in row {i} has exceeded the 150 character limit and is now deleted"
                )
                data.drop(i, inplace=True)
                data.reset_index(drop=True, inplace=True)
        else:
            print(f"The title in row {i} is not a string and is now deleted")
            data.drop(i, inplace=True)
            data.reset_index(drop=True, inplace=True)

    print("Checking titles done")
    return data


def check_tags(data):
    """
    Checks if the tags in the DataFrame are in the correct format
    """
    print("Checking tags")
    corpus = data["tags"].values
    for i in range(len(corpus)):
        tags = corpus[i]

        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str):
                    if len(tag) > 35:
                        print(
                            f"A tag in row {i} has exceeded the 35 character limit and is now deleted"
                        )
                        data.drop(i, inplace=True)
                        data.reset_index(drop=True, inplace=True)
                        break
                else:
                    print(f"A tag in row {i} is not a string and is now deleted")
                    data.drop(i, inplace=True)
                    data.reset_index(drop=True, inplace=True)
                    break

        else:
            print(f"Tags in row {i} is not a list and is now deleted")
            data.drop(i, inplace=True)
            data.reset_index(drop=True, inplace=True)
    print("Checking tags done")
    return data


def data_validation(data):
    """
    Validates the consistency and format of the data
    """
    print("--- DATA VALIDATION STARTED ---")
    cols = int(data.shape[1])
    if int(cols) == 2:
        data["tags"] = data["tags"].apply(literal_eval)
        column_check(data, cols)
        title_data = check_title(data)
        df = check_tags(title_data)
        return df

    elif int(cols) == 1:
        column_check(data, cols)
        df = check_title(data)
        return df

    else:
        print("Incorrect amount of columns")
        # TODO THROW EXCEPTION HERE
