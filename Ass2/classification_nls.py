import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from NeuralNetwork import *

if len(sys.argv) < 2:
    exit("Usage: python3 classification_nls.py <base_directory = .../Group21/>")

base = sys.argv[1]
if base[-1] != '/':
    base += "/"

##### NLS Classification #####
class_nls = "Classification/"
filename = "NLS_Group21.txt"
print()
print("Running NLS Classification")

(d,k) = (2,2)
path = base + class_nls + filename
df = pd.read_csv(path, sep="  ", names=["X1", "X2"], skiprows=1, engine='python')
total = 4892
X_dat = [np.asarray(df[:2446]), np.asarray(df[2446:4892])]
y_dat = [np.full((2446,k), Util.class_to_arr(k,0)), np.full((2446,k), Util.class_to_arr(k,1))]

# Actual class for each input data
plt.figure()
plt.plot(X_dat[0][:,0], X_dat[0][:,1], '.', label="Class 0")
plt.plot(X_dat[1][:,0], X_dat[1][:,1], '.', label="Class 1")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Actual classes")
plt.legend()

X_dat = np.reshape(X_dat, (total,d), 'F')
y_dat = np.reshape(y_dat, (total,k), 'F')

[(X_train, y_train), (X_valid, y_valid), (X_test, y_test)] = Util.split_data(X_dat, y_dat, 0.6, 0.2, 0.2)

# acc_max = -1
# for h1 in range(1, 26):
#     for h2 in range(1, 26):
#         f = FCNN([Layer(d, Util.logistic),
#                   Layer(h1, Util.logistic),
#                   Layer(h2, Util.logistic),
#                   Layer(k, Util.logistic)])
#         f.train(X_train, y_train, max_epoch=100, eta=0.9, optimize=True)
#         acc_valid = Util.accuracy(y_valid, f.test(X_valid))
#         print(f"Accuracy with {h1},{h2} neurons in the hidden layer: {acc_valid:.4}")
#         if acc_valid > acc_max:
#             print(f"Found optimal {h1},{h2} at {acc_valid:.4}")
#             (best_h1, best_h2) = (h1, h2)
#             acc_max = acc_valid
# print()

best_h1 = 14
best_h2 = 10
f = FCNN([Layer(d, Util.logistic),
          Layer(best_h1, Util.logistic),
          Layer(best_h2, Util.logistic),
          Layer(k, Util.logistic)])
print(f"Selecting {best_h1},{best_h2} neurons in the hidden layers")

err_train = f.train(X_train, y_train, optimize=True, debug=True)
plt.figure()
plt.plot(err_train)
plt.ylabel("MSE")
plt.xlabel("Number of epochs")

y_pred = f.test(X_test)
acc_test = Util.accuracy(y_test, y_pred)
confusion_mat = Util.confusion(y_test, y_pred)
print(f"Accuracy on test data: {(acc_test * 100):.2f}%")
print("Confusion matrix: ", confusion_mat, sep="\n")
print()

c0 = mpatches.Patch(color='red', label='Class 0')
c1 = mpatches.Patch(color='green', label='Class 1')

# Decision plot boundary
print("Plotting decision boundary")
grid = []
(X1_min, X2_min) = pd.DataFrame(X_test).min()
(X1_max, X2_max) = pd.DataFrame(X_test).max()
var = 5
for i in np.arange (X1_min - var, X1_max + var, 0.5):
    for j in np.arange(X2_min - var, X2_max + var, 0.5):
        grid.append([i,j])
grid = np.asarray(grid)

grid_pred = f.test(grid)
grid_pred = [Util.arr_to_class(grid_predi) for grid_predi in grid_pred]

plt.figure()
for i in range(len(grid)):
    if grid_pred[i] == 0:
        color = 'red'
    elif grid_pred[i] == 1:
        color = 'green'
    plt.plot(grid[i][0], grid[i][1], c=color, marker='.', alpha=0.1)

plt.plot(X_train[:,0], X_train[:,1], '.')
plt.legend(handles=[c0, c1])
plt.xlabel('X1')
plt.ylabel('X2')
plt.title("Decision plot boundary")

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
#     axs[0].append(figs[0].add_subplot(4, 4, h1+1, projection='3d'))
# for h2 in range(best_h2):
#     axs[1].append(figs[1].add_subplot(4, 4, h2+1, projection='3d'))
# for o in range(k):
#     axs[2].append(figs[2].add_subplot(1, k, o+1, projection='3d'))
# for ax in axs:
#     for i in range(len(ax)):
#         ax[i].set_title(f"Neuron {i+1}")
#         ax[i].set_xlabel("X1")
#         ax[i].set_ylabel("X2")
#         ax[i].set_zlabel("Activation")
#         ax[i].legend(handles=[c0, c1])
#
# y_pred = f.test(X_train)
# for i in range(len(X_train)):
#     pred = Util.arr_to_class(y_pred[i])
#     f.forward_compute(X_train[i])
#     if pred == 0:
#         color = 'red'
#     elif pred == 1:
#         color = 'green'
#     for h1 in range(best_h1):
#         axs[0][h1].scatter(X_train[i][0], X_train[i][1], f.network[1].A[h1], c=color, marker='.')
#     for h2 in range(best_h2):
#         axs[1][h2].scatter(X_train[i][0], X_train[i][1], f.network[2].A[h2], c=color, marker='.')
#     for o in range(k):
#         axs[2][o].scatter(X_train[i][0], X_train[i][1], f.network[3].A[o], c=color, marker='.')
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
#     axs[0].append(figs[0].add_subplot(4, 4, h1+1, projection='3d'))
# for h2 in range(best_h2):
#     axs[1].append(figs[1].add_subplot(4, 4, h2+1, projection='3d'))
# for o in range(k):
#     axs[2].append(figs[2].add_subplot(1, k, o+1, projection='3d'))
# for ax in axs:
#     for i in range(len(ax)):
#         ax[i].set_title(f"Neuron {i+1}")
#         ax[i].set_xlabel("X1")
#         ax[i].set_ylabel("X2")
#         ax[i].set_zlabel("Activation")
#         ax[i].legend(handles=[c0, c1])
#
# y_pred = f.test(X_valid)
# for i in range(len(X_valid)):
#     pred = Util.arr_to_class(y_pred[i])
#     f.forward_compute(X_valid[i])
#     if pred == 0:
#         color = 'red'
#     elif pred == 1:
#         color = 'green'
#     for h1 in range(best_h1):
#         axs[0][h1].scatter(X_valid[i][0], X_valid[i][1], f.network[1].A[h1], c=color, marker='.')
#     for h2 in range(best_h2):
#         axs[1][h2].scatter(X_valid[i][0], X_valid[i][1], f.network[2].A[h2], c=color, marker='.')
#     for o in range(k):
#         axs[2][o].scatter(X_valid[i][0], X_valid[i][1], f.network[3].A[o], c=color, marker='.')

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
    axs[0].append(figs[0].add_subplot(4, 4, h1+1, projection='3d'))
for h2 in range(best_h2):
    axs[1].append(figs[1].add_subplot(4, 4, h2+1, projection='3d'))
for o in range(k):
    axs[2].append(figs[2].add_subplot(1, k, o+1, projection='3d'))
for ax in axs:
    for i in range(len(ax)):
        ax[i].set_title(f"Neuron {i+1}")
        ax[i].set_xlabel("X1")
        ax[i].set_ylabel("X2")
        ax[i].set_zlabel("Activation")
        ax[i].legend(handles=[c0, c1])

y_pred = f.test(X_test)
for i in range(len(X_test)):
    pred = Util.arr_to_class(y_pred[i])
    f.forward_compute(X_test[i])
    if pred == 0:
        color = 'red'
    elif pred == 1:
        color = 'green'
    for h1 in range(best_h1):
        axs[0][h1].scatter(X_test[i][0], X_test[i][1], f.network[1].A[h1], c=color, marker='.')
    for h2 in range(best_h2):
        axs[1][h2].scatter(X_test[i][0], X_test[i][1], f.network[2].A[h2], c=color, marker='.')
    for o in range(k):
        axs[2][o].scatter(X_test[i][0], X_test[i][1], f.network[3].A[o], c=color, marker='.')


plt.show()
