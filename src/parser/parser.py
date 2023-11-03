
# open file and read it
import numpy as np
import os
import pickle


def get_unique_value(column_index, array):
    res = []
    for i in column_index:

        unique_values = np.unique(array[:, i])
        res.append(unique_values)
    return res


# custom something to compute the path in the name, to be able to save where we want
def parse_file(file_name, path_save="./data/output/temp/"):
    file_path, file_name = os.path.split(file_name)
    file_path += "/"
    if os.path.exists(path_save+file_name+"-array.pkl"):
        # File exists, so load the data from the file
        with open(path_save+file_name+"-array.pkl", 'rb') as file:
            array2d = pickle.load(file)
        print("File exists. Data has been loaded.")
    else:
        file = open(file_path+file_name, 'r')
        file_content = file.read()
        file_content = file_content.split('\n')
        res = []
        for line in file_content:
            line = line.split(',')
            if len(line) > 1:
                res.append(line)

        array = np.array(res, dtype=object)
        array2d = np.vstack(array)
        # File does not exist, so save the data to the file
        with open(path_save+file_name+"-array.pkl", 'wb') as file:
            pickle.dump(array2d, file)
        print("File does not exist. Data has been saved.")
    # set is now a 2d array

    return array2d


def parse_kdd(file_name, path_save):
    array = parse_file(file_name, path_save)
    index_symbol = [1, 2, 3, 6, 11, 20, 21]
    unique_values = get_unique_value(index_symbol, array)
    print(unique_values)
    # we now want to auto encode this


if __name__ == '__main__':
    a = parse_kdd('../../data/kddcup.data_10_percent',
                  "../../data/output/temp/")
