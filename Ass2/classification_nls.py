import sys
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

plt.figure()
plt.plot(X_dat[0][:,0], X_dat[0][:,1], 'o', label="Class 0")
plt.plot(X_dat[1][:,0], X_dat[1][:,1], 'o', label="Class 1")
plt.xlabel("X1")
plt.ylabel("X2")
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

err_train = f.train(X_train, y_train, optimize=True)
plt.figure()
plt.plot(err_train)
plt.ylabel("MSE")
plt.xlabel("Number of epochs")

y_pred = f.test(X_test)
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_test[:,0], X_test[:,1], [Util.arr_to_class(y_predi) for y_predi in y_pred])
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
acc_test = Util.accuracy(y_test, y_pred)
confusion_mat = Util.confusion(y_test, y_pred)
print(f"Accuracy on test data: {(acc_test * 100):.2f}%")
print("Confusion matrix: ", confusion_mat, sep="\n")


plt.show()
