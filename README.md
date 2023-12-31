# ML-F23-Project

This repository contains code for Parts I and II of our Machine Learning Final project and for our implmentation of the bonus part.

## Part I Data Preprocessing
The code we use for preprocessing our data is found in multiple files. First, we split our data into training and testing sets and write each set to a csv file in the file `create_train_val_test_split.py`. We encountered an issue with running dataset 2 on our models due to one of its features being in string form. Since the feature was binary (present/absent) we replace "present" and "absent" with the integer values 1 and 0, respectively. Code for this step can be found in `convert_data2_str_features_to_int.py`. Finally, we write the functions `get_data()` and `process_data()` in the file `run_all_models.py` to read the data from the csv files and format it to be input into our models. 

To run PCA and T-SNE analysis, please refer to the two file in `preprocessing` folder. The `utils.py` will contain all the functions that was used to generate the plots in our report. The name of it function explains it purpose. The file `dataViz.py` are used as example usage.

## How to Run Our Code for Part I
In Part I, we run Logistic Regression, Boosting, KNN, Decision Trees, Random Forest, and SVM. Our process consists of the following steps:

1) For each model, we establish a set of hyperparameters to test and run 10-fold cross-validation on each model using the training data for each dataset. We record the best classifier found for each model type. The code for these first two steps can be found in the file `run_all_models.py`. The best models found for each dataset can be found in the folder `output_metrics`.
2)  Next, we take the lists of parameters we eatsblished in step 1 and modify them as needed based on our results from cross-validation. Then, for each type of model, we train classifiers with all combinations of the hypoerparameters on all the training data and test on the test data. We measure the performance of each model with respect to the different hyperparameters considered to analyze how each hyperparameter contributes to the bias and variance of each mode. Code for this step can be found in the file `bias_variance_evaluation.py`. Outputs from this step can be found in the folder `bias_variance_comparisons`.
3)  Finally, we plot the train and test losses with respect to each hyperparameter considered for each model. Code for generating these plots can be found in the file `plot_results.py`. Outputs from this step can be found in the folder `plots`.

## How to Run Our Code Part II
In Part II, we run Multilayer Perceptron. All the codes could be located inside neural folder. There are two files:
- utils.py: all necessary function
- main.py: code execution

There is no step for processing except some manipulation to transpose and transform data into pytorch tensor.

1) To train model, use function `train_model()` which would require predefined set of parameter [model, epochs, loss_fun, optimizer, X_train, y_train]
2) To evaluate model on the training set, use function `eval_model_train()`, which would require a trained model, X_test, and y_test
3) To evaluate model on the testing set, use function `eval_model()`, which would require a trained model, X_test, y_test
4) To run parameter tuning, use function `unit_optimize()` which require predefinied set of parameter [epochs, loss_fun, X_train, y_train, X_test, y_test, unit1_range, unit2_range, lr_range]

To regenrate plot, please refer to `plots.Rmd`. All the plots and analysis in this part was done using R.

## How to Run code for Extra Credit
To run extra credit and perform an attack on reconstruction process, please refer to `xtracred` folder. The file `utils.py` include the class to generate a deep autoencoder that was created under the help of [this tutorial](https://www.geeksforgeeks.org/implement-deep-autoencoder-in-pytorch-for-image-reconstruction/). At the same time, `utils.py` also contain methods to calculate `frobenius_norm()`, `solver()` for optimization described in the project report.

The main.py file contain the content from the same GeeksForGeeks tutorial of how to train an autoencoder. The rest of the file are a demonstration of how everything was done to create an attack. Please refer to the comment in the file as instruction.
