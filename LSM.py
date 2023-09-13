import cv2 as cv
import numpy as np
from sklearn.manifold import TSNE
from read_mnist_dataset import x_test,x_train,y_test,y_train
from os.path  import join
import  matplotlib.pyplot as plt

hist_train = np.zeros((len(x_train),256))

for i in range(len(x_train)):
    for j in range(28):
        for k in range(28):
            hist_train[i,x_train[i][j][k]]+=1

np.savetxt("hist_train.csv", hist_train, delimiter=",")

hist_test = np.zeros((len(x_test),256))

for i in range(len(x_test)):
    for j in range(28):
        for k in range(28):
            hist_test[i,x_test[i][j][k]]+=1
            
np.savetxt("hist_test.csv", hist_test, delimiter=",")


plt.plot(hist_train)

# hist = cv.calcHist([img], [0], None, [256], [0, 256])

hist_train = np.loadtxt("hist_train.csv",delimiter=",")
hist_test = np.loadtxt("hist_test.csv",delimiter=',')

nlz_hist_train = hist_train.copy()
nlz_hist_test = hist_test.copy()

for i in range(hist_train.shape[0]):
    nlz_hist_train[i,:] = (hist_train[i,:] - min(hist_train[i,:]))/(max(hist_train[i,:])-min(hist_train[i,:]))
        
for i in range(hist_test.shape[0]):
    nlz_hist_test[i,:] = (hist_test[i,:] - min(hist_test[i,:]))/(max(hist_test[i,:])-min(hist_test[i,:]))

train_reduced = TSNE(n_components=2, verbose=1, random_state=42).fit_transform(nlz_hist_train[:1000,:])
test_reduced = TSNE(n_components=2, verbose=1, random_state=42).fit_transform(nlz_hist_test[:1000,:])

plt.figure(figsize=(10,8))
plt.scatter(train_reduced[:,0],train_reduced[:,1], c=y_train[:1000], cmap=plt.cm.get_cmap("tab10", 10), marker='^', s=8)
plt.colorbar(ticks=range(10))
plt.title("t-SNE plot of MNIST dataset")
plt.show()
# plt.scatter()

def LSM(x_test, x_train ,y_train,y_test):

  x_train = np.concatenate((np.ones((x_train.shape[0], 1), dtype=np.float64), x_train), axis=1)
  x_test = np.concatenate((np.ones((x_test.shape[0], 1), dtype=np.float64), x_test), axis=1)

  w = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train

  # Evaluating the model on the training data
  y_train_pred = x_train @ w
  mse_train = np.mean((y_train_pred - y_train) ** 2)

  # Evaluating the model on the testing data
  y_test_pred = x_test @ w
  mse_test = np.mean((y_test_pred - y_test) ** 2)
  print("Training MSE:", mse_train)
  print("Testing MSE:", mse_test)
  
LSM(test_reduced,train_reduced,y_train[:1000],y_test[:1000])
