import sys
from NeuralNetwork import *

if len(sys.argv) < 2:
    exit("Usage: python3 regression_bi.py <base_directory = .../Group21/>")

base = sys.argv[1]
if base[-1] != '/':
    base += "/"

##### Bivariate Regression #####
regression_bi = "Regression/BivariateData/21.csv"
print("Running Bivariate Regression")

(d,k) = (2,1)
path = base + regression_bi
df = pd.read_csv(path, names=["X1", "X2", "Y"])
(X_dat, y_dat) = (np.asarray(df[['X1', 'X2']]), np.asarray(df[['Y']]))

plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_dat[:,0], X_dat[:,1], y_dat[:,0])
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

[(X_train, y_train), (X_valid, y_valid), (X_test, y_test)] = Util.split_data(X_dat, y_dat, 0.6, 0.2, 0.2)

# err_min = np.inf
# for h1 in range(2,11):
#     for h2 in range(2, 11):
#         f = FCNN([Layer(d, Util.logistic),
#                   Layer(h1, Util.logistic),
#                   Layer(h2, Util.logistic),
#                   Layer(k, Util.linear)])

#         f.train(X_train, y_train, max_epoch=200)
#         err_valid = Util.error_avg(y_valid, f.test(X_valid))
#         print(f"MSE with {h1},{h2} neurons in the hidden layer: {err_valid:.4f}")
#         if err_valid < err_min:
#             print(f"Found optimal {h1},{h2} at {err_valid}")
#             (best_h1, best_h2) = (h1, h2)
#             err_min = err_valid
# print()

best_h1 = 5
best_h2 = 7
print(f"Selecting {best_h1},{best_h2} neurons in the hidden layer")
f = FCNN([Layer(d, Util.logistic),
          Layer(best_h1, Util.logistic),
          Layer(best_h2, Util.logistic),
          Layer(k, Util.linear)])

err_train = f.train(X_dat, y_dat, debug=True)
plt.figure()
plt.plot(err_train)
plt.ylabel("MSE")
plt.xlabel("Number of epochs")

y_pred = f.test(X_test)
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_test[:,0], X_test[:,1], np.asarray(y_pred)[:,0])
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
err_test = Util.error_avg(y_test, y_pred)
print(f"MSE on test data: {err_test:.4f}")


plt.show()
