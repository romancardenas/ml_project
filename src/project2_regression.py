import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
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

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = KFold(n_splits=K, shuffle=True)

# Initialize variables
Features = np.zeros((M, K))
Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_fs = np.empty((K, 1))
Error_test_fs = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))

k = 0
for train_index, test_index in CV.split(X):
    # extract training and test set for current CV fold
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]
    internal_cross_validation = 10

    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train - y_train.mean()).sum() / y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test - y_test.mean()).sum() / y_test.shape[0]

    # Compute squared error with all features selected (no feature selection)
    m = lm.LinearRegression(fit_intercept=True, normalize=True).fit(X_train, y_train)
    Error_train[k] = np.square(y_train - m.predict(X_train)).sum() / y_train.shape[0]
    Error_test[k] = np.square(y_test - m.predict(X_test)).sum() / y_test.shape[0]

    # Compute squared error with feature subset selection
    textout = 'verbose'
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,
                                                                          display=textout)
    Features[selected_features, k] = 1
    # .. alternatively you could use module sklearn.feature_selection
    if len(selected_features) is 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).')
    else:
        m = lm.LinearRegression(fit_intercept=True, normalize=True).fit(X_train[:, selected_features], y_train)
        Error_train_fs[k] = np.square(y_train - m.predict(X_train[:, selected_features])).sum() / y_train.shape[0]
        Error_test_fs[k] = np.square(y_test - m.predict(X_test[:, selected_features])).sum() / y_test.shape[0]

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
# Display results
print('\n')
print('Computing just the mean:\n')
print('- Training error: {0}'.format(Error_train_nofeatures.mean()))
print('- Test error:     {0}'.format(Error_test_nofeatures.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_nofeatures.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_nofeatures.sum())/Error_test_nofeatures.sum()))
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Linear regression with feature selection:\n')
print('- Training error: {0}'.format(Error_train_fs.mean()))
print('- Test error:     {0}'.format(Error_test_fs.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))

figure(k)
subplot(1, 3, 2)
bmplot(attributeNames, range(1, Features.shape[1]+1), -Features)
clim(-1.5, 0)
xlabel('Crossvalidation fold')
ylabel('Attribute')
show()
