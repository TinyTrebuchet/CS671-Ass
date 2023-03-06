import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Util:

    # Logistic activation function
    def logistic(x, derivative=False):
        fx = (1 / (1 + np.exp(-x)))
        if not derivative:
            return fx
        else:
            return (fx * (1 - fx))

    # ReLu
    def relu(x, derivative=False):
        if not derivative:
            return (x if x >= 0 else 0)
        else:
            return (1 if x >= 0 else 0)

    # Linear activation function
    def linear(x, derivative=False):
        if not derivative:
            return x
        else:
            return 1

    # Evaluate instantaneous error
    def error_inst(y, y_pred):
        sumv = 0
        for (y_k, y_pred_k) in zip(y, y_pred):
            sumv += (y_k - y_pred_k) ** 2
        return (sumv)

    # Evaluate average error
    def error_avg(y, y_pred):
        assert (len(y) == len(y_pred)), "Invalid data sizes"
        n = len(y)
        err_sum = 0
        for i in range(n):
            err_sum += Util.error_inst(y[i], y_pred[i])
        err_avg = err_sum / n
        return err_avg

    # Convert column matrix into row vector
    def col_to_arr(x_col):
        return np.asarray(x_col).flatten()

    # Convert row vector into column matrix
    def arr_to_col(x_vec):
        return np.asmatrix(x_vec).T

    # Convert class to array
    def class_to_arr(k, y):
        return [1 if i == y else 0 for i in range(k)]

    # Convert array to class
    def arr_to_class(arr):
        return (np.asarray(arr).argmax())

    # Shuffle the data together
    def shuffle(X_dat, y_dat):
        assert (len(X_dat) == len(y_dat)), "Invalid data sizes"
        n = len(X_dat)
        p = np.random.permutation(n)
        return (X_dat[p], y_dat[p])

    # Split data
    def split_data(X_dat, y_dat, *r):
        assert (sum(r) == 1), "Invalid distribution"
        ret = []
        n = len(X_dat)
        for ri in r:
            cr = int(ri * n)
            Xi, yi = X_dat[:cr], y_dat[:cr]
            X_dat, y_dat = X_dat[cr:], y_dat[cr:]
            ret.append((Xi, yi))
        return ret

    # Calculate accuracy
    def accuracy(y, y_pred):
        assert (len(y) == len(y_pred)), "Invalid data sizes"
        total = len(y)
        correct = 0
        for i in range(total):
            if (Util.arr_to_class(y[i]) == Util.arr_to_class(y_pred[i])):
                correct += 1
        accuracy = correct / total
        return accuracy

    # Generate confusion matrix
    def confusion(y, y_pred):
        assert (len(y) == len(y_pred)), "Invalid data sizes"
        n = len(y)
        k = len(y[0])
        confusion_mat = np.asmatrix(np.zeros((k,k), int))
        for i in range(n):
            confusion_mat[Util.arr_to_class(y[i]), Util.arr_to_class(y_pred[i])] += 1
        return confusion_mat

class Layer:
    def __init__(self, size, activation):
        assert (size > 0), "Invalid size"
        self.size = size
        self.activation = activation
        self.Z = np.empty(self.size)
        self.A = np.empty(self.size)

class FCNN:

    def __init__(self, layers):

        assert (len(layers) >= 2), "Atleast input and output layer should be given"
        self.nl = len(layers)
        self.d = layers[0].size
        self.k = layers[-1].size
        self.network = layers

        self.nW = self.nl - 1
        self.W_mats = []
        for i in range(self.nW):
            W_mat = np.asmatrix(np.random.rand(self.network[i].size + 1, self.network[i+1].size))
            self.W_mats.append(W_mat)
        self.ssg = [0]*self.nW
        self.epsilon = 1e-8

    # Forward computation
    def forward_compute(self, x_vec):
        self.network[0].A = x_vec
        for i in range(self.nW):
            Z_col = self.W_mats[i].T * np.vstack([1, Util.arr_to_col(self.network[i].A)])
            A_col = np.vectorize(self.network[i+1].activation)(Z_col)
            self.network[i+1].Z = Util.col_to_arr(Z_col)
            self.network[i+1].A = Util.col_to_arr(A_col)
        return self.network[-1].A

    # Compute deltas
    def compute_deltas(self, y_train):
        deltas = [[]] * self.nW
        for z in reversed(range(self.nW)):
            delta_vec = np.empty(self.network[z+1].size)
            for j in range(self.network[z+1].size):
                if (z == self.nW-1):
                    delta_vec[j] = -2 * (y_train[j] - self.network[z+1].A[j])
                else:
                    delta_vec[j] = 0
                    for k in range(self.network[z+2].size):
                        delta_vec[j] += deltas[z+1][k] * self.W_mats[z+1][j+1,k]
                delta_vec[j] *= self.network[z+1].activation(self.network[z+1].Z[j], derivative=True)
            deltas[z] = delta_vec
        return deltas

    # Backwards computation
    def backward_propogate(self, y_train, optimize=False):
        deltas = self.compute_deltas(y_train)
        for z in range(self.nW):
            gradient = (np.vstack([1, Util.arr_to_col(self.network[z].A)]) * np.asmatrix(deltas[z]))
            if optimize:
                self.ssg[z] += np.square(gradient)
                eta = self.eta / np.sqrt(self.ssg[z] + self.epsilon)
            else:
                eta = self.eta
            self.W_mats[z] -= np.multiply(eta, gradient)

    # Run 1 epoch
    def run_epoch(self, X_dat, y_dat=None, learn=False, optimize=False):
        assert (self.d == X_dat.shape[1]), "Invalid input data"
        if y_dat is not None:
            assert (self.k == y_dat.shape[1]), "Invalid input data"
            assert (len(X_dat) == len(y_dat)), "Invalid data sizes"

        n = len(X_dat)
        y_pred = [[]] * n
        for i in range(n):
            y_pred[i] = self.forward_compute(X_dat[i])
            if (learn):
                self.backward_propogate(y_dat[i], optimize)

        return y_pred

    # Train network on the training data
    def train(self, X_train, y_train, eta=None, threshold=0.001, max_epoch=400, optimize=False, debug=False):
        if eta is None:
            self.eta = 0.8 if optimize else 0.05
        else:
            self.eta = eta

        [(X_learn, y_learn), (X_valid, y_valid)] = Util.split_data(X_train, y_train, 0.9, 0.1)

        err_arr = []
        num_epoch = 0
        while (num_epoch <= max_epoch):
            num_epoch += 1
            (X_train, y_train) = Util.shuffle(X_train, y_train)

            self.run_epoch(X_learn, y_learn, learn=True, optimize=optimize)
            err_curr = Util.error_avg(y_valid, self.test(X_valid))
            err_arr.append(err_curr)
            if debug:
                print(f"Completed epoch {num_epoch} with MSE: {err_curr:.4f}")

            if err_curr < threshold:
                break

            if len(err_arr) > 6:
                similar = True
                for i in range(-2,-7,-1):
                    if abs(err_curr - err_arr[i]) > threshold/10:
                        similar = False
                        break
                if similar:
                    break

        return err_arr

    # Test network on test data
    def test(self, X_test):
        return self.run_epoch(X_test, learn=False)
