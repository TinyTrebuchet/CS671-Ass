import sys
import matplotlib.pyplot as plt
from NeuralNetwork import *

if len(sys.argv) < 2:
    exit("Usage: python3 regression_uni.py <base_directory = .../Group21/>")

base = sys.argv[1]
if base[-1] != '/':
    base += "/"

##### Univariate Regression #####
regression_uni = "Regression/UnivariateData/21.csv"
print("Running Univariate Regression")

(d,k) = (1,1)
path = base + regression_uni
df = pd.read_csv(path, names=["X", "Y"])
(X_dat, y_dat) = (np.asarray(df[['X']]), np.asarray(df[['Y']]))

# Actual output for each input data
plt.figure()
plt.plot(X_dat, y_dat, '.')
plt.xlabel('X')
plt.ylabel('Y')

[(X_train, y_train), (X_valid, y_valid), (X_test, y_test)] = Util.split_data(X_dat, y_dat, 0.6, 0.2, 0.2)

# err_min = np.inf
# for h in range(1,9):
#     f = FCNN([Layer(d, Util.logistic),
#               Layer(h, Util.logistic),
#               Layer(k, Util.linear)])
#     f.train(X_train, y_train)
#     err_valid = Util.error_avg(y_valid, f.test(X_valid))
#     print(f"MSE with {h} neurons in the hidden layer: {err_valid:.4f}")
#     if err_valid < err_min:
#         print(f"Found optimal {h} at {err_valid}")
#         best_h = h
#         err_min = err_valid
# print()

best_h = 3
print(f"Selecting {best_h} neurons in the hidden layer")
f = FCNN([Layer(d, Util.logistic),
          Layer(best_h, Util.logistic),
          Layer(k, Util.linear)])

err_train = f.train(X_train, y_train, optimize=True, debug=True)
plt.figure()
plt.plot(err_train)
plt.xlabel("Number of epochs")
plt.ylabel("MSE")

y_pred = f.test(X_test)
err_test = Util.error_avg(y_test, y_pred)
print(f"MSE on test data: {err_test:.4f}")

# Model ouput vs input for test data
plt.figure()
plt.plot(X_test, y_pred, '.')
plt.xlabel("X")
plt.ylabel("Y predicted")


plt.show()
