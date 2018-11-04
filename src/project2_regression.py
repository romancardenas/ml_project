import pandas as pd
import numpy as np
import neurolab as nl
from sklearn.model_selection import KFold
import sklearn.linear_model as lm
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from toolbox_02450 import feature_selector_lr, bmplot

# Read data from CSV file
data = pd.read_csv('../data/kc_house_data_clean_regression_nozip.csv')  # Select this if you don't want ZIP codes
# data = pd.read_csv('../data/kc_house_data_clean_regression_zip.csv')  # Select this if you want to include ZIP codes
# Make the script compatible with the course naming convention
attributeNames = list(data)[1:]
X = data.values[:, 1:]
y = data.values[:, 0]
N, M = X.shape
print(N)
print(M)

#################################################################################
#                                                                               #
#                             2-LAYER CROSS-VALIDATION                          #
#                                                                               #
#################################################################################

K = 5  # Outer cross-validation fold: 5-Fold
CV = KFold(n_splits=K, shuffle=True)

# Initialize variables for simple base case (just predicting always the mean)
Error_train_nofeatures = np.empty((K, 1))   # Train error (base case)
Error_test_nofeatures = np.empty((K, 1))    # Test error (base case)

# Initialize variables for linear regression
X_train_mean = np.zeros((K, M))             # It will contain the train set mean for each fold
X_train_std = np.ones((K, M))              # It will contain the train set standard deviation for each fold

LR_Error_train = np.empty((K, 1))           # Train error (all features)
LR_Error_test = np.empty((K, 1))            # Test error (all features)
LR_Features = np.zeros((M, K))              # For keeping track of selected features
LR_Error_train_fs = np.empty((K, 1))        # Train error (feature selection)
LR_Error_test_fs = np.empty((K, 1))         # Test error (feature selection)

# Initialize variables for artificial neural network
ANN_hidden_layers = np.empty((K, 1))        # Number of hidden layers (Artificial Neural Network)
ANN_Error_train = np.empty((K, 1))          # Train error (Artificial Neural Network)
ANN_Error_test = np.empty((K, 1))           # Test error (Artificial Neural Network)

n_hidden_units_test = [2, 3]                # number of hidden units to check
n_train = 2                                 # number of networks trained in each k-fold
learning_goal = 100000                      # stop criterion 1 (train mse to be reached)
max_epochs = 100                            # stop criterion 2 (max epochs in training)
show_error_freq = 5                         # frequency of training status updates

k = 0
for train_index, test_index in CV.split(X):  # Outer 2-layer cross-validation loop
    print("\n OUTER CROSS-VALIDATION {0}/{1}".format(k+1, K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}'.format(test_index))
    # extract training and test set for current CV fold
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]

    # Normalize the training set
    X_train_mean[k, :] = X_train.mean(axis=0)
    X_train_std[k, :] = X_train.std(axis=0)

    X_train = (X_train - X_train_mean[k, :]) / X_train_std[k, :]
    X_test = (X_test - X_train_mean[k, :]) / X_train_std[k, :]

    ##################################################################################
    #                                                                                #
    #                          BASE CASE (JUST PREDICTING THE MEAN                   #
    #                                                                                #
    ##################################################################################
    Error_train_nofeatures[k] = np.square(y_train - y_train.mean()).sum() / y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test - y_test.mean()).sum() / y_test.shape[0]

    ##################################################################################
    #                                                                                #
    #                     LINEAR REGRESSION USING ALL THE FEATURES                   #
    #                                                                                #
    ##################################################################################
    m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
    LR_Error_train[k] = np.square(y_train - m.predict(X_train)).sum() / y_train.shape[0]
    LR_Error_test[k] = np.square(y_test - m.predict(X_test)).sum() / y_test.shape[0]

    ##################################################################################
    #                                                                                #
    #                     LINEAR REGRESSION WITH FEATURE SELECTION                   #
    #                                                                                #
    ##################################################################################
    print('\nLINEAR REGRESSION MODEL')
    K_internal = 10
    textout = ''
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, K_internal, display=textout)
    LR_Features[selected_features, k] = 1
    if len(selected_features) is 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).')
    else:
        m = lm.LinearRegression(fit_intercept=True).fit(X_train[:, selected_features], y_train)
        LR_Error_train_fs[k] = np.square(y_train - m.predict(X_train[:, selected_features])).sum() / y_train.shape[0]
        LR_Error_test_fs[k] = np.square(y_test - m.predict(X_test[:, selected_features])).sum() / y_test.shape[0]

        figure(k)
        subplot(1, 2, 1)
        plot(range(1, len(loss_record)), loss_record[1:])
        xlabel('Iteration')
        ylabel('Squared error (crossvalidation)')

        subplot(1, 2, 2)
        bmplot(attributeNames, range(1, features_record.shape[1]), -features_record[:, 1:])
        clim(-1.5, 0)
        xlabel('Iteration')
        title('Linear regression CV: {0}'.format(k+1))
        show()
    print('Train error: {0}'.format(LR_Error_train[k]))
    print('Test error: {0}'.format(LR_Error_test[k]))
    print('Features no: {0}\n'.format(selected_features.size))

    ##################################################################################
    #                                                                                #
    #                           ARTIFICIAL NEURAL NETWORK                            #
    #                                                                                #
    ##################################################################################

    K_internal = 10
    n_hidden_units_select = np.empty((K_internal, 1))       # Store the selected number of hidden units
    inner_fold_error_ANN = np.empty((K_internal, 1))        # Store the error for each inner fold

    CV_ANN = KFold(K_internal, shuffle=True)
    inner_k = 0
    print("\n ARTIFICIAL NEURAL NETWORK")
    for ann_train_index, ann_test_index in CV_ANN.split(X_train):
        print("\n INNER CROSS-VALIDATION {0}/{1}".format(inner_k+1, K_internal))

        X_ANN_train = X_train[ann_train_index]
        y_ANN_train = y_train[ann_train_index]
        X_ANN_test = X_train[ann_test_index]
        y_ANN_test = y_train[ann_test_index]

        bestnet = None
        best_train_error = np.inf
        bestlayer = None
        for n_hidden_units in n_hidden_units_test:
            print("training with {0} hidden units...".format(n_hidden_units))
            for i in range(n_train):
                print('Training network {0}/{1}...'.format(i + 1, n_train))
                # Create randomly initialized network with 2 layers
                ann = nl.net.newff([[-3, 8]] * M, [n_hidden_units, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
                if i == 0:
                   bestnet = ann
                # train network
                train_error = ann.train(X_ANN_train, y_ANN_train.reshape(-1, 1), goal=learning_goal,
                                        epochs=max_epochs//2, show=show_error_freq)
                if train_error[-1] < best_train_error:
                    bestnet = ann
                    best_train_error = train_error[-1]
                    bestlayer = n_hidden_units
        print('Best train error: {0} (with {1} hidden layers)'.format(best_train_error, bestlayer))
        y_ANN_est = bestnet.sim(X_ANN_test).squeeze()
        inner_fold_error_ANN[k] = np.power(y_ANN_est - y_ANN_test, 2).sum().astype(float) / y_ANN_test.shape[0]
        n_hidden_units_select[k] = bestlayer
        inner_k += 1
    best_index = np.argmin(inner_fold_error_ANN)
    ANN_hidden_layers[k] = n_hidden_units_select[best_index]
    best_ann = nl.net.newff([[-3, 8]] * M, [n_hidden_units_select[best_index], 1], [nl.trans.TanSig(), nl.trans.PureLin()])
    bestnet = None
    best_train_error = np.inf
    for i in range(n_train):
        if i == 0:
            bestnet = ann
        error = best_ann.train(X_train, y_train.reshape(-1, 1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
        if train_error[-1] < best_train_error:
            bestnet = ann
            best_train_error = train_error[-1]
    ANN_Error_train[k] = best_train_error
    y_est = bestnet.sim(X_test).squeeze()
    ANN_Error_test[k] = np.power(y_test - y_est, 2).sum().astype(float) / y_test.shape[0]
    print('Train error: {0}'.format(ANN_Error_train[k]))
    print('Test error: {0}'.format(ANN_Error_test[k]))
    k += 1

print("\n")
print('##################################################################################')
print('#                                                                                #')
print('#                                   DISPLAY RESULTS                              #')
print('#                                                                                #')
print('##################################################################################')
print('\n')
print('Computing just the mean (base case):')
print('- Training error: {0}'.format(Error_train_nofeatures.mean()))
print('- Test error:     {0}'.format(Error_test_nofeatures.mean()))
print('\n')
print('------------------------------- LINEAR REGRESSION --------------------------------')
print('\n')
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(LR_Error_train.mean()))
print('- Test error:     {0}'.format(LR_Error_test.mean()))
print('- R^2 train:      {0}'.format((Error_train_nofeatures.sum() - LR_Error_train.sum()) / Error_train_nofeatures.sum()))
print('- R^2 test:       {0}'.format((Error_test_nofeatures.sum() - LR_Error_test.sum()) / Error_test_nofeatures.sum()))
print('\n')
print('Linear regression with feature selection:')
print('- Training error: {0}'.format(LR_Error_train_fs.mean()))
print('- Test error:     {0}'.format(LR_Error_test_fs.mean()))
print('- R^2 train:      {0}'.format((Error_train_nofeatures.sum() - LR_Error_train_fs.sum()) / Error_train_nofeatures.sum()))
print('- R^2 test:       {0}'.format((Error_test_nofeatures.sum() - LR_Error_test_fs.sum()) / Error_test_nofeatures.sum()))
print('\n')
print('---------------------------- ARTIFICIAL NEURAL NETWORK ----------------------------')
print('\n')
print('- Number of hidden layer for each fold: {0}'.format(str(ANN_hidden_layers)))
print('- Training error: {0}'.format(ANN_Error_train.mean()))
print('- Test error: {0}'.format(ANN_Error_test.mean()))
print('- R^2 train:      {0}'.format((Error_train_nofeatures.sum() - LR_Error_train_fs.sum()) / Error_train_nofeatures.sum()))
print('- R^2 test:       {0}'.format((Error_test_nofeatures.sum() - LR_Error_test_fs.sum()) / Error_test_nofeatures.sum()))
