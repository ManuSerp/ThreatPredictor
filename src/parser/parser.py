"""parser code"""
# open file and read it
import os
import pickle
import numpy as np
from tqdm import tqdm

def get_unique_value(column_index, array):
    res = []
    for i in column_index:

        unique_values = np.unique(array[:, i])
        res.append(unique_values)
    return res


# custom something to compute the path in the name, to be able to save where we want
def parse_file(file_name, path_save="../data/output/temp/"):
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
        file.close()
        file_content = file_content.split('\n')
        res = []
        for line in file_content:
            line = line.split(',')
            if len(line) > 1:
                res.append(line)

        array = np.array(res)
        array2d = np.vstack(array)
        # File does not exist, so save the data to the file
        with open(path_save+file_name+"-array.pkl", 'wb') as file:
            pickle.dump(array2d, file)
        print("File does not exist. Data has been saved.")
    # set is now a 2d array

    return array2d


def one_hot_encode(array, index_symbol,log_path="../data/output/parsing/",tag=''): 
    # class better too much args
    unique_values = get_unique_value(index_symbol, array)
    with open(log_path+tag+"_ohe_log.txt","w") as file:
        for x, i in enumerate(index_symbol):
            file.write("Column "+str(i)+" has "+str(len(unique_values[x]))+" unique values.\n")
            file.write("Unique values are: "+str(unique_values[x])+"\n")
            file.write("\n")

    newarray = array.copy()
    print("Creating new array...")
    for x, i in enumerate(index_symbol):
        for _ in tqdm(range(len(unique_values[x]))):
            newarray = np.insert(newarray, i, 0, axis=1) 
            # completement faux, besoin de se rapeller des index qui augmentent, faut faire en ordre decroissant l'insertion
        #penser a supprimer la colonne d'avant
    print("Assigning bool...")
    for id in tqdm(range(len(array))):
        ligne=array[id]
        for x, i in enumerate(index_symbol):
            pos=unique_values[x].tolist().index(ligne[i])
            newarray[id][i+pos]=1
    # we now have a one hot encoded array
    return newarray

def get_ohe(array, index_symbol,log_path,path_save, tag=''): # class better too much args
    str_save=path_save+"_ohe"+tag+".pkl"
    if os.path.exists(str_save):
        print("Ohe File detected!")
        # File exists, so load the data from the file
        with open(str_save, 'rb') as file:
            array_out = pickle.load(file)
        print("Ohe "+path_save+"ohe"+tag+".pkl"+"exists. Data has been loaded.")

    else:
        print("Ohe File not detected! building it...")
        array_out=one_hot_encode(array, index_symbol,log_path)
        with open(str_save, 'wb') as file:
            pickle.dump(array_out,file)
        print("Ohe "+path_save+"ohe"+tag+".pkl"+"does not exist. Data has been saved.")

    
    return array_out   

    


def parse_kdd(file_name, path_save, log_path):
    array = parse_file(file_name, path_save) # ca ne devrait pasa etre appelé si on a le pkl de ohe
    index_symbol = [1, 2, 3, 6, 11, 20, 21]
    array_out = get_ohe(array, index_symbol,log_path,path_save)
    # we now want to auto encode this
    print(array_out[0])
    return array_out


if __name__ == '__main__':
    PATH_SAVE="../../data/output/temp/"
    LOG_PATH="../../data/output/parsing/"
    a = parse_kdd('../../data/kddcup.data_10_percent', PATH_SAVE,LOG_PATH)
