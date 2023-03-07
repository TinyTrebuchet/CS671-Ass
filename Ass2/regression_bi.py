import sys
import matplotlib.pyplot as plt
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

# Actual output for each input data
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X_dat[:,0], X_dat[:,1], y_dat[:,0], marker='.')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title("Actual output")

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

print()

# Target Output and Model Output vs Data Plot
fig = plt.figure()

ax = fig.add_subplot(2, 3, 1, projection='3d')
ax.scatter3D(X_train[:,0], X_train[:,1], np.asarray(y_train)[:,0])
ax.set_title("Target Output VS Train Data")
ax.set_xlabel('X1_train')
ax.set_ylabel('X2_train')
ax.set_zlabel('Target Output')

ax = fig.add_subplot(2, 3, 2, projection='3d')
ax.scatter3D(X_valid[:,0], X_valid[:,1], np.asarray(y_valid)[:,0])
ax.set_title("Target Output VS Validation Data")
ax.set_xlabel('X1_Validation')
ax.set_ylabel('X2_Validation')
ax.set_zlabel('Target Output')

ax = fig.add_subplot(2, 3, 3, projection='3d')
ax.scatter3D(X_test[:,0], X_test[:,1], np.asarray(y_test)[:,0])
ax.set_title("Target Output VS Test Data")
ax.set_xlabel('X1_test')
ax.set_ylabel('X2_test')
ax.set_zlabel('Target Output')

ax = fig.add_subplot(2, 3, 4, projection='3d')
ax.scatter3D(X_train[:,0], X_train[:,1], np.asarray(y_pred_train)[:,0])
ax.set_title("Model Output VS Train Data")
ax.set_xlabel('X1_train')
ax.set_ylabel('X2_train')
ax.set_zlabel('Model Output')

ax = fig.add_subplot(2, 3, 5, projection='3d')
ax.scatter3D(X_valid[:,0], X_valid[:,1], np.asarray(y_pred_validation)[:,0])
ax.set_title("Model Output VS Validation Data")
ax.set_xlabel('X1_Validation')
ax.set_ylabel('X2_Validation')
ax.set_zlabel('Model Output')

ax = fig.add_subplot(2, 3, 6, projection='3d')
ax.scatter3D(X_test[:,0], X_test[:,1], np.asarray(y_pred_test)[:,0])
ax.set_title("Model Output VS Test Data")
ax.set_xlabel('X1_test')
ax.set_ylabel('X2_test')
ax.set_zlabel('Model Output')


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

# # Activation for each neuron vs input for training data
# print("Plotting activation for each neuron vs input for training data")
# figs = [plt.figure() for _ in range(3)]
# for i in range(len(figs)):
#     if i == len(figs)-1:
#         figs[i].suptitle("Activation for each neuron on training data [Output layer]")
#     else:
#         figs[i].suptitle(f"Activation for each neuron on training data [Hidden layer {i+1}]")
#
# axs = [[] for _ in range(3)]
# for h1 in range(best_h1):
#     axs[0].append(figs[0].add_subplot(3, 3, h1+1, projection='3d'))
# for h2 in range(best_h2):
#     axs[1].append(figs[1].add_subplot(3, 3, h2+1, projection='3d'))
# for o in range(k):
#     axs[2].append(figs[2].add_subplot(1, k, o+1, projection='3d'))
# for ax in axs:
#     for i in range(len(ax)):
#         ax[i].set_title(f"Neuron {i+1}")
#         ax[i].set_xlabel("X1")
#         ax[i].set_ylabel("X2")
#         ax[i].set_zlabel("Activation")
#
# y_pred = f.test(X_train)
# for i in range(len(X_train)):
#     pred = y_pred[i]
#     f.forward_compute(X_train[i])
#     for h1 in range(best_h1):
#         axs[0][h1].scatter(X_train[i][0], X_train[i][1], f.network[1].A[h1], c='blue', marker='.')
#     for h2 in range(best_h2):
#         axs[1][h2].scatter(X_train[i][0], X_train[i][1], f.network[2].A[h2], c='blue', marker='.')
#     for o in range(k):
#         axs[2][o].scatter(X_train[i][0], X_train[i][1], f.network[3].A[o], c='blue', marker='.')
#
# # Activation for each neuron vs input for validation data
# print("Plotting activation for each neuron vs input for validation data")
# figs = [plt.figure() for _ in range(3)]
# for i in range(len(figs)):
#     if i == len(figs)-1:
#         figs[i].suptitle("Activation for each neuron on validation data [Output layer]")
#     else:
#         figs[i].suptitle(f"Activation for each neuron on validation data [Hidden layer {i+1}]")
#
# axs = [[] for _ in range(3)]
# for h1 in range(best_h1):
#     axs[0].append(figs[0].add_subplot(3, 3, h1+1, projection='3d'))
# for h2 in range(best_h2):
#     axs[1].append(figs[1].add_subplot(3, 3, h2+1, projection='3d'))
# for o in range(k):
#     axs[2].append(figs[2].add_subplot(1, k, o+1, projection='3d'))
# for ax in axs:
#     for i in range(len(ax)):
#         ax[i].set_title(f"Neuron {i+1}")
#         ax[i].set_xlabel("X1")
#         ax[i].set_ylabel("X2")
#         ax[i].set_zlabel("Activation")
#
# y_pred = f.test(X_valid)
# for i in range(len(X_valid)):
#     pred = y_pred[i]
#     f.forward_compute(X_valid[i])
#     for h1 in range(best_h1):
#         axs[0][h1].scatter(X_valid[i][0], X_valid[i][1], f.network[1].A[h1], c='blue', marker='.')
#     for h2 in range(best_h2):
#         axs[1][h2].scatter(X_valid[i][0], X_valid[i][1], f.network[2].A[h2], c='blue', marker='.')
#     for o in range(k):
#         axs[2][o].scatter(X_valid[i][0], X_valid[i][1], f.network[3].A[o], c='blue', marker='.')

# Activation for each neuron vs input for test data
print("Plotting activation for each neuron vs input for test data")
figs = [plt.figure() for _ in range(3)]
for i in range(len(figs)):
    if i == len(figs)-1:
        figs[i].suptitle("Activation for each neuron on test data [Output layer]")
    else:
        figs[i].suptitle(f"Activation for each neuron on test data [Hidden layer {i+1}]")

axs = [[] for _ in range(3)]
for h1 in range(best_h1):
    axs[0].append(figs[0].add_subplot(3, 3, h1+1, projection='3d'))
for h2 in range(best_h2):
    axs[1].append(figs[1].add_subplot(3, 3, h2+1, projection='3d'))
for o in range(k):
    axs[2].append(figs[2].add_subplot(1, k, o+1, projection='3d'))
for ax in axs:
    for i in range(len(ax)):
        ax[i].set_title(f"Neuron {i+1}")
        ax[i].set_xlabel("X1")
        ax[i].set_ylabel("X2")
        ax[i].set_zlabel("Activation")

y_pred = f.test(X_test)
for i in range(len(X_test)):
    pred = y_pred[i]
    f.forward_compute(X_test[i])
    for h1 in range(best_h1):
        axs[0][h1].scatter(X_test[i][0], X_test[i][1], f.network[1].A[h1], c='blue', marker='.')
    for h2 in range(best_h2):
        axs[1][h2].scatter(X_test[i][0], X_test[i][1], f.network[2].A[h2], c='blue', marker='.')
    for o in range(k):
        axs[2][o].scatter(X_test[i][0], X_test[i][1], f.network[3].A[o], c='blue', marker='.')


plt.show()
