import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv

def plot_train_and_test_errors(model_name, model_attributes_dict, training_errors, testing_errors):

    sample_attribute = list(model_attributes_dict.keys())[0]
    print(sample_attribute)
    print(model_attributes_dict[sample_attribute])
    num_attributes = len(model_attributes_dict[sample_attribute])

    train_test_distinctions = ["train" for i in range(num_attributes)] + ["test" for i in range(num_attributes)]
    errors = training_errors + testing_errors

    for model_attribute in model_attributes_dict.keys():

        attribute_vals = model_attributes_dict[model_attribute] 
        attribute_vals = attribute_vals + attribute_vals  #duplicate

        x_axis_name = model_name + " " + model_attribute + " Value"

        df = pd.DataFrame({
            x_axis_name: attribute_vals,
            "Error": training_errors + testing_errors,
            "Error Type": train_test_distinctions 
        })

        plt.figure()
        error_pointplot = sns.pointplot(data=df, x=x_axis_name, y="Error", hue="Error Type")
        plt.savefig("plots/" + model_name.replace(" ", "_") + "_" + model_attribute + ".jpg")
        plt.savefig("plots/" + model_name.replace(" ", "_") + "_" + model_attribute + ".svg")

def plot_decision_tree_errors(decision_trees_csv_path, dataset_name:str):
    decision_trees_csv_file = open(decision_trees_csv_path, newline='')
    csvreader = csv.reader(decision_trees_csv_file)

    #get values for the different model attributes to be tested
    max_depths = []
    min_samples_splits = []
    min_samples_leafs = []

    training_errors = []
    testing_errors = []

    first_row = True 

    for row in csvreader:

        if first_row == True:
            first_row = False
            continue #skip the first row since it contains names of attributes

        train_error = float(row[6])
        test_error = float(row[7])

        max_depth = int(row[8])
        min_samples_split = int(row[9])
        min_samples_leaf = int(row[10])

        training_errors.append(train_error)
        testing_errors.append(test_error)

        max_depths.append(max_depth)
        min_samples_splits.append(min_samples_split)
        min_samples_leafs.append(min_samples_leaf)

    model_attributes_dict = {}
    model_attributes_dict ["Max Depth"] = max_depths
    model_attributes_dict ["Min Samples Split"] = min_samples_splits
    model_attributes_dict ["Min Samples Leaf"] = min_samples_leafs

    plot_train_and_test_errors("Decision Tree " + dataset_name, model_attributes_dict, training_errors, testing_errors)


def plot_boosting_errors(boosting_csv_path, dataset_name:str):
    pass

def plot_logistic_regression_errors(logistic_regression_csv_path, dataset_name:str):
    logistic_regression_csv_file = open(logistic_regression_csv_path, newline='')
    csvreader = csv.reader(logistic_regression_csv_file)

    #get values for the different model attributes to be tested
    penalties = []
    max_iters = []

    training_errors = []
    testing_errors = []

    first_row = True 

    for row in csvreader:

        if first_row == True:
            first_row = False
            continue #skip the first row since it contains names of attributes

        train_error = float(row[6])
        test_error = float(row[7])

        penalty = row[8]
        max_iter = int(row[9])

        training_errors.append(train_error)
        testing_errors.append(test_error)

        penalties.append(penalty)
        max_iters.append(max_iter)

    model_attributes_dict = {}
    model_attributes_dict ["Penalty"] = penalties
    model_attributes_dict ["Max Iter"] = max_iters

    plot_train_and_test_errors("Logistic Regression " + dataset_name, model_attributes_dict, training_errors, testing_errors)

def plot_svm_errors(svm_csv_path, dataset_name:str):
    svm_csv_file = open(svm_csv_path, newline='')
    csvreader = csv.reader(svm_csv_file)

    #get values for the different model attributes to be tested
    C = []
    kernel = []
    gamma = []

    training_errors = []
    testing_errors = []

    first_row = True 

    for row in csvreader:

        if first_row == True:
            first_row = False
            continue #skip the first row since it contains names of attributes

        train_error = float(row[6])
        test_error = float(row[7])

        C_ = float(row[8])
        kernel_ = str(row[9])
        gamma_ = float(row[10])

        training_errors.append(train_error)
        testing_errors.append(test_error)

        C.append(C_)
        kernel.append(kernel_)
        gamma.append(gamma_)

    model_attributes_dict = {}
    model_attributes_dict ["C"] = C
    model_attributes_dict ["Kernel"] = kernel
    model_attributes_dict ["Gamma"] = gamma

    plot_train_and_test_errors("SVM " + dataset_name, model_attributes_dict, training_errors, testing_errors)


def plot_kfold_knn_errors(kfold_knn_csv_path, dataset_name:str):
    kfold_random_forest_csv_file = open(kfold_random_forest_csv_path, newline='')
    csvreader = csv.reader(kfold_random_forest_csv_file)

    n_neighbors_list = []

    training_errors = []
    testing_errors = []

    for row in csvreader:

        if first_row == True:
            first_row = False
            continue #skip the first row since it contains names of attributes

        train_error = float(row[6])
        test_error = float(row[7])

        n_neighbors = int(row([8]))

        training_errors.append(train_error)
        testing_errors.append(test_error)

        n_neighbors_list.append(n_neighbors)

        model_attributes_dict = {}
        modeL_attributes_dict["Number of Neighbors"]



def plot_kfold_random_forest_errors(kfold_random_forest_csv_path, dataset_name:str):
    kfold_random_forest_csv_file = open(kfold_random_forest_csv_path, newline='')
    csvreader = csv.reader(kfold_random_forest_csv_file)

    #get values for the different model attributes to be tested
    n_estimators_list = []
    max_depths = []

    training_errors = []
    testing_errors = []

    first_row = True 

    for row in csvreader:

        if first_row == True:
            first_row = False
            continue #skip the first row since it contains names of attributes

        train_error = float(row[6])
        test_error = float(row[7])

        n_estimators = int(row[8])
        max_depth = int(row[9])

        training_errors.append(train_error)
        testing_errors.append(test_error)

        n_estimators_list.append(n_estimators)
        max_depths.append(max_depth)

    model_attributes_dict = {}
    model_attributes_dict ["Number of Estimators"] = n_estimators_list
    model_attributes_dict ["Max Depth"] = max_depths

    plot_train_and_test_errors("K-fold Random Forest " + dataset_name, model_attributes_dict, training_errors, testing_errors)




# decision_tree_dataset1_csv = "bias_variance_comparisons/decision_tree_dataset1.csv"
# plot_decision_tree_errors(decision_tree_dataset1_csv, "Dataset 1")

# decision_tree_dataset2_csv = "bias_variance_comparisons/decision_tree_dataset2.csv"
# plot_decision_tree_errors(decision_tree_dataset2_csv, "Dataset 2")

# logistic_regression_dataset1_csv = "bias_variance_comparisons/logistic_regression_dataset1.csv"
# plot_logistic_regression_errors(logistic_regression_dataset1_csv, "Dataset 1")

# logistic_regression_dataset2_csv = "bias_variance_comparisons/logistic_regression_dataset2.csv"
# plot_logistic_regression_errors(logistic_regression_dataset2_csv, "Dataset 2")

#svm_dataset1_csv = "bias_variance_comparisons/svm_dataset1.csv"
#plot_svm_errors(svm_dataset1_csv, "Dataset 1")

#svm_dataset2_csv = "bias_variance_comparisons/svm_dataset2.csv"
#plot_svm_errors(svm_dataset2_csv, "Dataset 2")

# kfold_random_forest_dataset1_csv = "bias_variance_comparisons/kfold_random_forest_dataset1.csv"
# plot_kfold_random_forest_errors(kfold_random_forest_dataset1_csv, "Dataset 1")

kfold_random_forest_dataset2_csv = "bias_variance_comparisons/kfold_random_forest_dataset2.csv"
plot_kfold_random_forest_errors(kfold_random_forest_dataset2_csv, "Dataset 2")




    
