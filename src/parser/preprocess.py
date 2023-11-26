"""preprocess code"""
from sklearn.utils import shuffle
import csv


class Preprocess:
    def __init__(self, data_path) -> None:
        self.data_path = data_path
        self.labels_dict = dict()

    def create_label_content(self):
        file = open(self.data_path, 'r')
        file_content = file.read()
        file.close()
        file_content = file_content.split('\n')
        index = 0
        for line in file_content:
            line = line.split(',')
            if (line != ''):
                label = line[-1]
                if label in self.labels_dict:
                    self.labels_dict[label].append(index)
                else:
                    self.labels_dict[label] = [index]
                index += 1
    
    def print_labels_dict(self):
        print("=== Data information ===\n")
        max_label_length = max(len(label) for label in self.labels_dict.keys())
        print(f"{'Label'.ljust(max_label_length)} | {'Length'}")
        print(f"{'-' * max_label_length} | {'-' * 6}")

        for label, list in self.labels_dict.items():
            print(f"{label.ljust(max_label_length)} | {len(list)}")

    
    def save_labels_dict(self,csv_path):
        with open(csv_path,'w') as file:
            writer=csv.writer(file, delimiter=',',lineterminator='\n',)
            writer.writerow(['Label', 'List'])
            for label,list in self.labels_dict.items():
                writer.writerow([label, list])
    
    def create_train_test_dataset(self, train_dataset_path,test_dataset_path, clustering_length, ratio):
        # Calculating the initial average length of data per label for the train + test set
        remaining_labels = len(self.labels_dict)
        remaining_length = clustering_length
        len_by_label = remaining_length / remaining_labels

        train_indices = []
        test_indices = []

        # Sort the labels by the length of their lists
        sorted_items = sorted(self.labels_dict.items(), key=lambda x: len(x[1]))
        self.labels_dict = dict(sorted_items)

        for label, data_list in self.labels_dict.items():
            # Shuffle the list before splitting
            shuffled_list = shuffle(data_list, random_state=0)

            train_len_by_label = ratio * len_by_label
            test_len_by_label = (1-ratio) * len_by_label

            # Calculate the number of items to include in train and test for this label
            num_train = min(len(data_list), int(train_len_by_label))
            num_test = min(len(data_list) - num_train, int(test_len_by_label))

            # Split the list into training and testing
            train_indices.extend(shuffled_list[:num_train])
            test_indices.extend(shuffled_list[num_train:num_train + num_test])

            # Update length for the next label
            remaining_length -= (num_train + num_test)
            remaining_labels -= 1
            len_by_label = remaining_length / remaining_labels if remaining_labels > 0 else 0

            # Correct  arround division error with the last label to have exactly length_clustering data
            last_indix = num_train + num_test
            while (remaining_labels == 0) and (len(train_indices) + len(test_indices) < clustering_length):
                remaining_ratio = len(train_indices) / (len(train_indices) + len(test_indices))
                if remaining_ratio > ratio:
                    test_indices.extend(shuffled_list[last_indix:last_indix+1])
                else:
                    train_indices.extend(shuffled_list[last_indix:last_indix+1])
        
        print(len(test_indices), "+", len(train_indices),"=",len(train_indices)+len(test_indices))

        self._write_dataset_to_csv(train_indices, train_dataset_path)
        self._write_dataset_to_csv(test_indices, test_dataset_path)
    
    def _write_dataset_to_csv(self, indices_list, file_path):
        with open(self.data_path, 'r') as file:
            lines = file.readlines()
        
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            for index in indices_list:
                try:
                    file.write(lines[index])
                except IndexError:
                    print("\033[93m!!WARNING!!\033[0m") 
                    print("Index out of range: ", index)
                    print("Lines length: ", len(lines))
                    

if __name__ == "__main__":
    DATA_PATH = "../../data/kddcup.data_10_percent"
    TRAIN_DATASET_PATH = "../../data/output/temp/train.csv"
    TEST_DATASET_PATH = "../../data/output/temp/test.csv"
    preprocess = Preprocess(DATA_PATH)
    preprocess.create_label_content()
    preprocess.print_labels_dict()
    preprocess.create_train_test_dataset(TRAIN_DATASET_PATH,TEST_DATASET_PATH,100000,0.8)
        

