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

# Model ouput vs input for test data
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X_test[:,0], X_test[:,1], [Util.arr_to_class(y_predi) for y_predi in y_pred], marker='.')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

# Activation for each neuron vs input for test data
fig = plt.figure()
axs = []
c0 = mpatches.Patch(color='red', label='Class 0')
c1 = mpatches.Patch(color='green', label='Class 1')
c2 = mpatches.Patch(color='blue', label='Class 2')
for h in range(best_h):
    axs.append(fig.add_subplot(1, best_h, h+1, projection='3d'))
    axs[-1].set_title(f"Neuron {h+1}")
    axs[-1].set_xlabel("X1")
    axs[-1].set_ylabel("X2")
    axs[-1].set_zlabel("Activation")
    axs[-1].legend(handles=[c0, c1, c2])

for i in range(len(X_test)):
    pred = Util.arr_to_class(y_pred[i])
    f.forward_compute(X_test[i])
    for h in range(best_h):
        if pred == 0:
            color = 'red'
        elif pred == 1:
            color = 'green'
        elif pred == 2:
            color = 'blue'
        axs[h].scatter(X_test[i][0], X_test[i][1], f.network[1].A[h], c=color, marker='.')

plt.show()
