from mnist import MNIST
import numpy as np

# load dataset
# download files from http://yann.lecun.com/exdb/mnist/
mndata = MNIST('./')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())
X_train = X_train/255.0
X_test = X_test/255.0
labels_train = labels_train.astype(float)
labels_test = labels_test.astype(float)

# for breaking apart the data, see the following
# boolean indexing:
labels_train == 7
# logical operations on arrays
np.logical_or
