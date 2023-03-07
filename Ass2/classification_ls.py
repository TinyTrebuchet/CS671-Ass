import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from NeuralNetwork import *

if len(sys.argv) < 2:
    exit("Usage: python3 classification_ls.py <base_directory = .../Group21/>")

base = sys.argv[1]
if base[-1] != '/':
    base += "/"

##### LS Classification #####
class_ls = "Classification/LS_Group21/"
print()
print("Running LS Classification")

total = 0
X_dat = []
y_dat = []
(d,k) = (2,3)
for (y, filename) in enumerate(["Class1.txt", "Class2.txt", "Class3.txt"]):
    path = base + class_ls + filename
    df = pd.read_csv(path, sep=" ", names=["X1", "X2"])
    total += len(df)
    X_dat.append(np.asarray(df))
    y_dat.append(np.full((len(df), k), Util.class_to_arr(k,y)))

# Actual class for each input data
plt.figure()
plt.plot(X_dat[0][:,0], X_dat[0][:,1], '.', label="Class 0")
plt.plot(X_dat[1][:,0], X_dat[1][:,1], '.', label="Class 1")
plt.plot(X_dat[2][:,0], X_dat[2][:,1], '.', label="Class 2")
plt.xlabel('X1')
plt.ylabel('X2')
plt.title("Actual classes")
plt.legend()

X_dat = np.reshape(X_dat, (total,d), 'F')
y_dat = np.reshape(y_dat, (total,k), 'F')

[(X_train, y_train), (X_valid, y_valid), (X_test, y_test)] = Util.split_data(X_dat, y_dat, 0.6, 0.2, 0.2)

# acc_max = -1
# for h in range(1, 9):
#     f = FCNN([Layer(d, Util.logistic),
#               Layer(h, Util.logistic),
#               Layer(k, Util.logistic)])
#     f.train(X_train, y_train, optimize=True, max_epoch=100, debug=True)
#     acc_valid = Util.accuracy(y_valid, f.test(X_valid))
#     print(f"Accuracy with {h} neurons in the hidden layer: {(acc_valid * 100):.2f}%")
#     if acc_valid > acc_max:
#         print(f"Found optimial {h} at {acc_valid}")
#         best_h = h
#         acc_max = acc_valid
# print()

best_h = 3
print(f"Selecting {best_h} neurons in the hidden layer")
f = FCNN([Layer(d, Util.logistic),
          Layer(best_h, Util.logistic),
          Layer(k, Util.logistic)])

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
c2 = mpatches.Patch(color='blue', label='Class 2')


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
    elif grid_pred[i] == 2:
        color = 'blue'
    plt.plot(grid[i][0], grid[i][1], c=color, marker='.', alpha=0.1)

plt.plot(X_train[:,0], X_train[:,1], '.')
plt.legend(handles=[c0, c1, c2])
plt.xlabel('X1')
plt.ylabel('X2')
plt.title("Decision plot boundary")

# # Activation for each neuron vs input for training data
# print("Plotting activation for each neuron vs input for training data")
# figs = [plt.figure() for _ in range(2)]
# for i in range(len(figs)):
#     if i == len(figs)-1:
#         figs[i].suptitle("Activation for each neuron on training data [Output layer]")
#     else:
#         figs[i].suptitle(f"Activation for each neuron on training data [Hidden layer {i+1}]")
#
# axs = [[] for _ in range(2)]
# for h in range(best_h):
#     axs[0].append(figs[0].add_subplot(1, best_h, h+1, projection='3d'))
# for o in range(k):
#     axs[1].append(figs[1].add_subplot(1, k, o+1, projection='3d'))
# for ax in axs:
#     for i in range(len(ax)):
#         ax[i].set_title(f"Neuron {i+1}")
#         ax[i].set_xlabel("X1")
#         ax[i].set_ylabel("X2")
#         ax[i].set_zlabel("Activation")
#         ax[i].legend(handles=[c0, c1, c2])
#
#
# y_pred = f.test(X_train)
# for i in range(len(X_train)):
#     pred = Util.arr_to_class(y_pred[i])
#     f.forward_compute(X_train[i])
#     if pred == 0:
#         color = 'red'
#     elif pred == 1:
#         color = 'green'
#     elif pred == 2:
#         color = 'blue'
#     for h in range(best_h):
#         axs[0][h].scatter(X_train[i][0], X_train[i][1], f.network[1].A[h], c=color, marker='.')
#     for o in range(k):
#         axs[1][o].scatter(X_train[i][0], X_train[i][1], f.network[2].A[o], c=color, marker='.')
#
# # Activation for each neuron vs input for validation data
# print("Plotting activation for each neuron vs input for validation data")
# figs = [plt.figure() for _ in range(2)]
# for i in range(len(figs)):
#     if i == len(figs)-1:
#         figs[i].suptitle("Activation for each neuron on validation data [Output layer]")
#     else:
#         figs[i].suptitle(f"Activation for each neuron on validation data [Hidden layer {i+1}]")
#
# axs = [[] for _ in range(2)]
# for h in range(best_h):
#     axs[0].append(figs[0].add_subplot(1, best_h, h+1, projection='3d'))
# for o in range(k):
#     axs[1].append(figs[1].add_subplot(1, k, o+1, projection='3d'))
# for ax in axs:
#     for i in range(len(ax)):
#         ax[i].set_title(f"Neuron {i+1}")
#         ax[i].set_xlabel("X1")
#         ax[i].set_ylabel("X2")
#         ax[i].set_zlabel("Activation")
#         ax[i].legend(handles=[c0, c1, c2])
#
#
# y_pred = f.test(X_valid)
# for i in range(len(X_valid)):
#     pred = Util.arr_to_class(y_pred[i])
#     f.forward_compute(X_valid[i])
#     if pred == 0:
#         color = 'red'
#     elif pred == 1:
#         color = 'green'
#     elif pred == 2:
#         color = 'blue'
#     for h in range(best_h):
#         axs[0][h].scatter(X_valid[i][0], X_valid[i][1], f.network[1].A[h], c=color, marker='.')
#     for o in range(k):
#         axs[1][o].scatter(X_valid[i][0], X_valid[i][1], f.network[2].A[o], c=color, marker='.')

# Activation for each neuron vs input for test data
print("Plotting activation for each neuron vs input for test data")
figs = [plt.figure() for _ in range(2)]
for i in range(len(figs)):
    if i == len(figs)-1:
        figs[i].suptitle("Activation for each neuron on test data [Output layer]")
    else:
        figs[i].suptitle(f"Activation for each neuron on test data [Hidden layer {i+1}]")

axs = [[] for _ in range(2)]
for h in range(best_h):
    axs[0].append(figs[0].add_subplot(1, best_h, h+1, projection='3d'))
for o in range(k):
    axs[1].append(figs[1].add_subplot(1, k, o+1, projection='3d'))
for ax in axs:
    for i in range(len(ax)):
        ax[i].set_title(f"Neuron {i+1}")
        ax[i].set_xlabel("X1")
        ax[i].set_ylabel("X2")
        ax[i].set_zlabel("Activation")
        ax[i].legend(handles=[c0, c1, c2])


y_pred = f.test(X_test)
for i in range(len(X_test)):
    pred = Util.arr_to_class(y_pred[i])
    f.forward_compute(X_test[i])
    if pred == 0:
        color = 'red'
    elif pred == 1:
        color = 'green'
    elif pred == 2:
        color = 'blue'
    for h in range(best_h):
        axs[0][h].scatter(X_test[i][0], X_test[i][1], f.network[1].A[h], c=color, marker='.')
    for o in range(k):
        axs[1][o].scatter(X_test[i][0], X_test[i][1], f.network[2].A[o], c=color, marker='.')


plt.show()
