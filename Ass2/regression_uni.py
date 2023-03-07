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
plt.plot(X_dat, y_dat, 'o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Actual output")

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


#############################

y_pred_train = f.test(X_train)
err_train = Util.error_avg(y_train, y_pred_train)
print(f"MSE on train data: {err_train:.4f}")

y_pred_validation = f.test(X_valid)
err_validation = Util.error_avg(y_valid, y_pred_validation)
print(f"MSE on validation data: {err_validation:.4f}")

y_pred_test = f.test(X_test)
err_test = Util.error_avg(y_test, y_pred_test)
print(f"MSE on test data: {err_test:.4f}")

# Target Output and Model Output vs Data Plot
plt.figure()

plt.subplot(2,3,1)
plt.plot(X_train, y_train, 'o')
plt.grid()
plt.title('Target Output VS Train Data')
plt.xlabel('X_Train')
plt.ylabel('Y_Train')

plt.subplot(2,3,2)
plt.plot(X_valid, y_valid, 'o')
plt.grid()
plt.title('Target Output VS Validation Data')
plt.xlabel('X_Validation')
plt.ylabel('Y_Validation')

plt.subplot(2,3,3)
plt.plot(X_test, y_test, 'o')
plt.grid()
plt.title('Target Output VS Test Data')
plt.xlabel('X_Test')
plt.ylabel('Y_Test')

plt.subplot(2,3,4)
plt.plot(X_train, y_pred_train, 'x')
plt.grid()
plt.title('Model Output VS Train Data')
plt.xlabel('X_Train')
plt.ylabel('Y_Pred_Train')

plt.subplot(2,3,5)
plt.plot(X_valid, y_pred_validation, 'x')
plt.grid()
plt.title('Model Output VS Validation Data')
plt.xlabel('X_Validation')
plt.ylabel('Y_Pred_Validation')

plt.subplot(2,3,6)
plt.plot(X_test, y_pred_test, 'x')
plt.grid()
plt.title('Model Output VS Test Data')
plt.xlabel('X_Test')
plt.ylabel('Y_Pred_Test')

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)


# Scatter Plot -  Model Output (Y) vs Target Output (X)
plt.figure()

plt.subplot(1,3,1)
plt.scatter(y_train,y_pred_train)
plt.xlabel('Target Output')
plt.ylabel('Model Output')
plt.title('Train Data')

plt.subplot(1,3,2)
plt.scatter(y_valid,y_pred_validation)
plt.xlabel('Target Output')
plt.ylabel('Model Output')
plt.title('Validation Data')

plt.subplot(1,3,3)
plt.scatter(y_test,y_pred_test)
plt.xlabel('Target Output')
plt.ylabel('Model Output')
plt.title('Test Data')

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)


plt.show()
