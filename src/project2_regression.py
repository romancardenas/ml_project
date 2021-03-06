import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import sklearn.linear_model as lm
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim, legend
from toolbox_02450 import feature_selector_lr, bmplot
from scipy import stats

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

# Read data from CSV file
#data = pd.read_csv('../data/kc_house_data_clean_regression_nozip.csv')  # Select this if you don't want ZIP codes
data = pd.read_csv('../data/kc_house_data_clean_regression_zip.csv')  # Select this if you want to include ZIP codes
# Make the script compatible with the course naming convention
attributeNames = list(data)[1:]
X = data.values[:, 1:]
y = data.values[:, 0]

print(attributeNames)

# Standardize data
X_mean = X.mean(axis=0)  # Store mean and standard deviation for data reconstruction
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

N, M = X.shape

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
LR_Error_train = np.empty((K, 1))           # Train error (all features)
LR_Error_test = np.empty((K, 1))            # Test error (all features)
LR_Features_fs = np.zeros((M, K))           # For keeping track of selected features
LR_Params_fs = list()                       # List with linear regression params
LR_Error_train_fs = np.empty((K, 1))        # Train error (feature selection)
LR_Error_test_fs = np.empty((K, 1))         # Test error (feature selection)

# Initialize variables for artificial neural network
ANN_hidden_units = np.empty((K, 1))         # Number of hidden units (Artificial Neural Network)
ANN_Error_train = np.empty((K, 1))          # Train error (Artificial Neural Network)
ANN_Error_test = np.empty((K, 1))           # Test error (Artificial Neural Network)

adam = optimizers.Adam(lr=0.3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)

n_hidden_units_test = [22, 24, 26, 28]         # number of hidden units to check (multiplied by the number of inputs)
n_hidden_units_test = [i * M for i in n_hidden_units_test]
n_train = 2                                 # number of networks trained in each k-fold
learning_goal = 100000                      # stop criterion 1 (train mse to be reached)
max_epochs = 1000                           # stop criterion 2 (max epochs in training)
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
    LR_Features_fs[selected_features, k] = 1
    m = lm.LinearRegression(fit_intercept=True).fit(X_train[:, selected_features], y_train)
    LR_Params_fs.append(m.coef_)

    LR_Error_train_fs[k] = np.square(y_train - m.predict(X_train[:, selected_features])).sum() / y_train.shape[0]
    y_est = m.predict(X_test[:, selected_features])
    LR_Error_test_fs[k] = np.square(y_test - m.predict(X_test[:, selected_features])).sum() / y_test.shape[0]

    figure()
    plot(y_test, y_est)
    title('Linear regression with forward feature selection')
    xlabel('Real values')
    ylabel('Estimated values')
    show()

    figure()
    plot(range(1, len(loss_record)), loss_record[1:])
    xlabel('Iteration')
    ylabel('Squared error (crossvalidation)')
    title('Linear regression with forward feature selection CV: {0}/{1}'.format(k + 1, K))
    show()

    print('Train error: {0}'.format(LR_Error_train_fs[k]))
    print('Test error: {0}'.format(LR_Error_test_fs[k]))
    print('Features no: {0}\n'.format(selected_features.size))

    ##################################################################################
    #                                                                                #
    #                           ARTIFICIAL NEURAL NETWORK                            #
    #                                                                                #
    ##################################################################################
    K_internal = 3
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
        best_val_error = np.inf
        bestlayer = None
        for n_hidden_units in n_hidden_units_test:
            print("training with {0} hidden units...".format(n_hidden_units))
            for i in range(n_train):
                print('Training network {0}/{1}...'.format(i + 1, n_train))

                model = Sequential()
                model.add(Dense(n_hidden_units, input_shape=(M,)))
                model.add(Activation('relu'))
                model.add(Dense(1))
                model.add(Activation('linear'))
                model.compile(loss='mean_squared_error', optimizer=adam)
                history = model.fit(X_ANN_train, y_ANN_train, epochs=max_epochs//2, batch_size=X_ANN_train.shape[0],
                                    verbose=0, validation_data=(X_ANN_test, y_ANN_test))

                train_error = history.history['loss'][-1]
                val_error = history.history['val_loss'][-1]
                print('        Training error: {0}'.format(train_error))
                if train_error < best_train_error:
                    bestnet = model
                    best_train_error = train_error
                    best_val_error = val_error
                    bestlayer = n_hidden_units
        print('Best train error: {0} (with {1} hidden layers)'.format(best_train_error, bestlayer))
        print('Best validation error: {0} (with {1} hidden layers)'.format(best_val_error, bestlayer))
        inner_fold_error_ANN[inner_k] = best_val_error
        n_hidden_units_select[inner_k] = bestlayer
        inner_k += 1
    best_index = np.asscalar(np.argmin(inner_fold_error_ANN))
    ANN_hidden_units[k] = n_hidden_units_select[best_index]

    bestnet = None
    best_train_error = np.inf
    best_val_error = np.inf
    for i in range(n_train):
        model = Sequential()
        aux = np.asscalar(n_hidden_units_select[best_index])
        model.add(Dense(int(np.asscalar(n_hidden_units_select[best_index])), input_shape=(M,)))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('linear'))
        model.compile(loss='mean_squared_error', optimizer=adam)

        history = model.fit(X_train, y_train, epochs=max_epochs, batch_size=X_train.shape[0], verbose=0,
                            validation_data=(X_test, y_test))
        train_error = history.history['loss'][-1]
        val_error = history.history['val_loss'][-1]
        print('        Training error: {0}'.format(train_error))
        if train_error < best_train_error:
            bestnet = model
            best_train_error = train_error
            best_val_error = val_error
    ANN_Error_train[k] = best_train_error
    ANN_Error_test[k] = best_val_error
    k += 1

# Figure with ANN generalization error
figure(figsize=(9, 6))
plot(range(1, ANN_Error_test.shape[0]+1), ANN_Error_test)
xlabel('Iteration')
ylabel('Squared error (crossvalidation)')
title('Artificial Neural Network Generalization Error')
show()


# Figure with linear regression with feature selection generalization error
figure(figsize=(9, 6))
plot(range(1, LR_Error_test_fs.shape[0]+1), LR_Error_test_fs)
xlabel('Iteration')
ylabel('Squared error (crossvalidation)')
title('Linear Regression with Feature Selection Generalization Error')
show()

# Figure with everything
figure(figsize=(9, 6))
plot(range(1, Error_test_nofeatures.shape[0]+1), Error_test_nofeatures)
plot(range(1, LR_Error_test_fs.shape[0]+1), LR_Error_test_fs)
plot(range(1, ANN_Error_test.shape[0]+1), ANN_Error_test)
legend(['Average', 'LR with fs', 'ANN'])
xlabel('Iteration')
ylabel('Squared error (crossvalidation)')
title('Models Generalization Error')
show()


print('----------------------------------- FOR REPORT -----------------------------------')
print('- [ANN] Number of hidden units for each fold: {0}'.format(str(ANN_hidden_units)))
print('- [ANN] Generalization error for each fold: {0}'.format(str(ANN_Error_test)))
print('- [LR_fs] Generalization error for each fold: {0}'.format(str(LR_Error_test_fs)))

for i in range(LR_Features_fs.shape[1]):
    used_attributes = [attributeNames[j] for j in range(LR_Features_fs.shape[0]) if LR_Features_fs[j, i] > 0]
    used_params = {used_attributes[j]: LR_Params_fs[i][j] for j in range(len(used_attributes))}
    print('- [LR_fs] feature selection for fold {0}: {1}'.format(i + 1, str(used_attributes)))
    print('- [LR_fs] parameters used (with normalized data): {0}'.format(str(used_params)))
print("DATA SET MEAN: {0}".format(X_mean))
print("DATA SET STANDARD DEVIATION: {0}".format(X_std))

print("\n\n\n\n\n")
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

print('- Training error: {0}'.format(ANN_Error_train.mean()))
print('- Test error: {0}'.format(ANN_Error_test.mean()))
print('- R^2 train:      {0}'.format((Error_train_nofeatures.sum() - ANN_Error_train.sum()) / Error_train_nofeatures.sum()))
print('- R^2 test:       {0}'.format((Error_test_nofeatures.sum() - ANN_Error_test.sum()) / Error_test_nofeatures.sum()))

print('\n')
print('-------------------------------- MODELS COMPARISON --------------------------------')
print('\n')
z = (LR_Error_test_fs - ANN_Error_test)
zb = z.mean()
nu = K - 1
sig = (z-zb).std() / np.sqrt(K-1)
alpha = 0.05
zL = zb + sig * stats.t.ppf(alpha/2, nu)
zH = zb + sig * stats.t.ppf(1-alpha/2, nu)
if zL <= 0 and 0 <= zH:
    print('Classifiers are not significantly different')
else:
    print('Classifiers are significantly different.')
