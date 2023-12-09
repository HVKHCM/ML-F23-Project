# ML-F23-Project

This repository contains code for Parts I and II of our Machine Learning Final project and for our implmentation of the bonus part.

## Data Preprocessing
The code we use for preprocessing our data is found in multiple files. First, we split our data into training and testing sets and write each set to a csv file in the file `create_train_val_test_split.py`. We encountered an issue with running dataset 2 on our models due to one of its features being in string form. Since the feature was binary (present/absent) we replace "present" and "absent" with the integer values 1 and 0, respectively. Code for this step can be found in `convert_data2_str_features_to_int.py`. Finally, we write the functions `get_data()` and `process_data()` in the file `run_all_models.py` to read the data from the csv files and format it to be input into our models. 

## How to Run Our Code for Part I
In Part I, we run Logistic Regression, Boosting, KNN, Decision Trees, Random Forest, and SVM. Our process consists of the following steps:

1) For each model, we establish a set of hyperparameters to test and run 10-fold cross-validation on each model using the training data for each dataset. We record the best classifier found for each model type. The code for these first two steps can be found in the file `run_all_models.py`. The best models found for each dataset can be found in the folder `output_metrics`.
2)  Next, we take the lists of parameters we eatsblished in step 1 and modify them as needed based on our results from cross-validation. Then, for each type of model, we train classifiers with all combinations of the hypoerparameters on all the training data and test on the test data. We measure the performance of each model with respect to the different hyperparameters considered to analyze how each hyperparameter contributes to the bias and variance of each mode. Code for this step can be found in the file `bias_variance_evaluation.py`. Outputs from this step can be found in the folder `bias_variance_comparisons`.
3)  Finally, we plot the train and test losses with respect to each hyperparameter considered for each model. Code for generating these plots can be found in the file `plot_results.py`. Outputs from this step can be found in the folder `plots`.
