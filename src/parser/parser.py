"""parser code"""
# open file and read it
import os
import pickle
import math
import numpy as np
from tqdm import tqdm


def int_to_binary_list(n):
    if n == 0:
        return [0]

    binary_string = bin(n)[2:] 
    binary_list = [int(bit) for bit in binary_string]

    return binary_list

def get_unique_value(column_index, array):
    res = []
    for i in column_index:

        unique_values = np.unique(array[:, i])
        res.append(unique_values)
    return res

def int_cleaning(array):
    label = array[:, -1]
    array = array[:, :-1]

    array=array.astype(np.float64)
    print(array.dtype)
    return array, label


class Parser:
    def __init__(self,path_save="../data/output/temp/",log_path="../data/output/parsing/"):
        self.path_save = path_save
        self.log_path = log_path
        self.array = None
        self.tag = ''
    
    


    def parse_file(self,file_name):
        file_path, file_name = os.path.split(file_name)
        file_path += "/"
        if os.path.exists(self.path_save+file_name+"-array.pkl"):
            # File exists, so load the data from the file
            with open(self.path_save+file_name+"-array.pkl", 'rb') as file:
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
            with open(self.path_save+file_name+"-array.pkl", 'wb') as file:
                pickle.dump(array2d, file)
            print("File does not exist. Data has been saved.")
        # set is now a 2d array

        return array2d


    def one_hot_encode(self,array, index_symbol): 
        unique_values = get_unique_value(index_symbol, array)
        with open(self.log_path+self.tag+"_ohe_log.txt","w") as file:
            for x, i in enumerate(index_symbol):
                file.write("Column "+str(i)+" has "+str(len(unique_values[x]))+" unique values.\n")
                file.write("Unique values are: "+str(unique_values[x])+"\n")
                file.write("\n")

        newarray = array.copy()
        print("Creating new array...")
        for x, i in enumerate(index_symbol[::-1]):
            newarray = np.delete(newarray, i, axis=1)
            longueur=len(unique_values[len(index_symbol)-x-1])
            binary_length= math.log(longueur,2)
            bl=math.ceil(binary_length)
            for _ in tqdm(range(bl)):
                newarray = np.insert(newarray, i, 0, axis=1) 
        print("Assigning binary bool value...")
        for id in tqdm(range(len(array))):
            ligne=array[id]
            for x, i in enumerate(index_symbol):
                pos=unique_values[x].tolist().index(ligne[i])
                binary_list=int_to_binary_list(pos) # binary encoding
                for index, value in enumerate(binary_list):

                    newarray[id][i+index]=value
        # we now have a one hot encoded array binary
        return newarray

    def get_ohe(self,array, index_symbol): 

        str_save=self.path_save+"_ohe"+self.tag+".pkl"
        if os.path.exists(str_save):
            print("Ohe File detected!")
            # File exists, so load the data from the file
            with open(str_save, 'rb') as file:
                array_out = pickle.load(file)
            print("Ohe "+self.path_save+"ohe"+self.tag+".pkl"+"exists. Data has been loaded.")

        else:
            print("Ohe File not detected! building it...")
            array_out=self.one_hot_encode(array, index_symbol)
            with open(str_save, 'wb') as file:
                pickle.dump(array_out,file)
            print("Ohe "+self.path_save+"ohe"+self.tag+".pkl"+"does not exist. Data has been saved.")

        
        return array_out   

    def set_tag(self,tag):
        self.tag=tag



## KDD CUP 99

    def parse_kdd(self, file_name):
        array = self.parse_file(file_name) # ca ne devrait pasa etre appelé si on a le pkl de ohe
        index_symbol = [1, 2, 3, 6, 11, 20, 21]
        array_out = self.get_ohe(array, index_symbol)
        array_out, label = int_cleaning(array_out) 
        sparsity = 1.0 - np.count_nonzero(array_out) / array_out.size 
        print("Sparsity of the array is: ")
        print(sparsity)
        return array_out,label


if __name__ == '__main__':
    PATH_SAVE="../../data/output/temp/"
    LOG_PATH="../../data/output/parsing/"
    parser=Parser(PATH_SAVE,LOG_PATH)
    a = parser.parse_kdd('../../data/kddcup.data_10_percent')
    print(a[0])
