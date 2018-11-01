import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import sklearn.linear_model as lm
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from toolbox_02450 import feature_selector_lr, bmplot

# Read data from CSV file
data = pd.read_csv('../data/kc_house_data_clean_regression.csv')
# Make the script compatible with the course naming convention
attributeNames = list(data)[2:]
X = data.values[:, 1:]
y = data.values[:, 0]
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
LR_Features = np.zeros((M, K))              # For keeping track of selected features
LR_Error_train_fs = np.empty((K, 1))        # Train error (feature selection)
LR_Error_test_fs = np.empty((K, 1))         # Test error (feature selection)



##################################################################################
#                                                                                #
#                                  LINEAR REGRESSION                             #
#                                                                                #
##################################################################################
print("\n")
print('STARTING WITH LINEAR REGRESSION MODEL')
k = 0
for train_index, test_index in CV.split(X): # Outer 2-layer cross-validation loop
    # extract training and test set for current CV fold
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]

    # BASE CASE (JUST PREDICTING THE MEAN)
    Error_train_nofeatures[k] = np.square(y_train - y_train.mean()).sum() / y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test - y_test.mean()).sum() / y_test.shape[0]

    # USING ALL THE FEATURES
    m = lm.LinearRegression(fit_intercept=True, normalize=True).fit(X_train, y_train)
    LR_Error_train[k] = np.square(y_train - m.predict(X_train)).sum() / y_train.shape[0]
    LR_Error_test[k] = np.square(y_test - m.predict(X_test)).sum() / y_test.shape[0]

    # USING FEATURE SELECTION
    K_internal = 10
    # textout = 'verbose'
    textout = ''
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, K_internal, display=textout)
    LR_Features[selected_features, k] = 1
    if len(selected_features) is 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).')
    else:
        m = lm.LinearRegression(fit_intercept=True, normalize=True).fit(X_train[:, selected_features], y_train)
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
        show()
    print('Cross validation fold {0}/{1}'.format(k + 1, K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}'.format(test_index))
    print('Features no: {0}\n'.format(selected_features.size))
    k += 1
print('FINISHING WITH LINEAR REGRESSION MODEL')
print("\n")

##################################################################################
#                                                                                #
#                                   DISPLAY RESULTS                              #
#                                                                                #
##################################################################################
print('##################################################################################')
print('#                                                                                #')
print('#                                   DISPLAY RESULTS                              #')
print('#                                                                                #')
print('##################################################################################')
print('------------------------------- LINEAR REGRESSION --------------------------------')
print('\n')
print('Computing just the mean (base case):')
print('- Training error: {0}'.format(Error_train_nofeatures.mean()))
print('- Test error:     {0}'.format(Error_test_nofeatures.mean()))
print('\n')
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(LR_Error_train.mean()))
print('- Test error:     {0}'.format(LR_Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum() - LR_Error_train.sum()) / Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum() - LR_Error_test.sum()) / Error_test_nofeatures.sum()))
print('\n')
print('Linear regression with feature selection:')
print('- Training error: {0}'.format(LR_Error_train_fs.mean()))
print('- Test error:     {0}'.format(LR_Error_test_fs.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum() - LR_Error_train_fs.sum()) / Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum() - LR_Error_test_fs.sum()) / Error_test_nofeatures.sum()))

figure(k)
subplot(1, 3, 2)
bmplot(attributeNames, range(1, LR_Features.shape[1] + 1), -LR_Features)
clim(-1.5, 0)
xlabel('Crossvalidation fold')
ylabel('Attribute')
show()

# Inspect selected feature coefficients effect on the entire dataset and
# plot the fitted model residual error as function of each attribute to
# inspect for systematic structure in the residual
# TODO revisar esto... Las etiquetas no cuadran
f = 2  # cross-validation fold to inspect
ff = LR_Features[:, f - 1].nonzero()[0]
if len(ff) is 0:
    print('\nNo features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).')
else:
    m = lm.LinearRegression(fit_intercept=True, normalize=True).fit(X[:, ff], y)

    y_est = m.predict(X[:, ff])
    residual = y - y_est

    figure(k + 1, figsize=(12, 6))
    title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
    for i in range(0, len(ff)):
        subplot(2, np.ceil(len(ff) / 2.0), i + 1)
        plot(X[:, ff[i]], residual, '.')
        xlabel(attributeNames[ff[i]])
        ylabel('residual error')
    show()
