import pandas as pd
import csv
import random 

def create_train_val_test_split(data_csv_path:str, dataset_name:str, train_split_percentage:float, val_split_percentage:float, test_split_percentage:float):
    data_csv_file = open(data_csv_path, newline='')
    data_csv_reader = csv.reader(data_csv_file)

    train_data_csv = open(dataset_name + "_training_data.csv", 'w', newline='')
    train_data_writer = csv.writer(train_data_csv)

    val_data_csv = open(dataset_name + "_validation_data.csv", 'w', newline='')
    val_data_writer = csv.writer(val_data_csv)

    test_data_csv = open(dataset_name + "_testing_data.csv", 'w', newline='')
    test_data_writer = csv.writer(test_data_csv)

    row_list = []

    for row in data_csv_reader:
        row_list.append(row)

    num_examples = len(row_list)
    train_split = round(train_split_percentage * num_examples) #https://docs.python.org/3/library/functions.html?highlight=round#round
    val_split = round(val_split_percentage * num_examples)
    test_split = round(test_split_percentage * num_examples)

    train_index = train_split
    val_index = train_index + val_split
    test_index = num_examples

    num_samples_after_division = train_split + val_split + test_split

    if num_samples_after_division != num_examples:
        num_examples_needed = num_examples - num_samples_after_division
        if num_examples_needed < 0:
            raise ValueError("You should not have more examples than are in the list.")

        train_index += num_examples_needed 

    random.shuffle(row_list) #https://docs.python.org/3/library/random.html

    train_examples = row_list[0:train_index]
    val_examples = row_list[train_index:val_index]
    test_examples = row_list[val_index:test_index]

    assert len(train_examples) + len(val_examples) + len(test_examples) == num_examples

    for row in train_examples:
        train_data_writer.writerow(row)
    
    for row in val_examples:
        val_data_writer.writerow(row)

    for row in test_examples:
        test_data_writer.writerow(row)


    
dataset_1_csv_path = "data1.csv"
dataset_1_name = "dataset1"

dataset_2_csv_path = "data2_with_all_numerical_features.csv"
dataset_2_name = 'dataset2'

train_split_percentage = 0.8
val_split_percentage = 0.0
test_split_percentage = 0.2


#Create train/val/test splits for each dataset
#create_train_val_test_split(dataset_1_csv_path, dataset_1_name, train_split_percentage, val_split_percentage, test_split_percentage)
create_train_val_test_split(dataset_2_csv_path, dataset_2_name, train_split_percentage, val_split_percentage, test_split_percentage)






