# coding=utf-8

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.metrics.classification import accuracy_score
import numpy as np

from dbn import SupervisedDBNClassification


# Loading dataset
digits = load_digits()
X, Y = digits.data, digits.target

# Data scaling
X = (X / 16).astype(np.float32)

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.1,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         l2_regularization=0.0,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)

# Test
Y_pred = classifier.predict(X_test)

print('Done.\nAccuracy: ')
print(accuracy_score(Y_test, Y_pred))