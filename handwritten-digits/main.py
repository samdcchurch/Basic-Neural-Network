# type: ignore
from tensorflow.keras.datasets import mnist
import numpy as np

from utils.activation_functions import sigmoid, relu
from utils.loss_functions import BCE

# STEP 1: DEFINE HYPER-PARAMETERS
n = [784, 16, 16, 10]
g = [None, relu, relu, sigmoid]
Loss = BCE
learning_rate = 0.15
num_iterations = 1000
loss_notification = 100


# STEP 2: IMPORT THE DATA
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# STEP 3: FORMAT THE DATA, INITIALIZE PARAMETERS
m = train_images.shape[0]
assert n[0] == train_images.shape[1] * train_images.shape[2]
L = len(n) - 1

X = train_images.reshape(m, n[0]).T / 255
Y = np.zeros((10, m))
for i, label in enumerate(train_labels):
    Y[label, i] = 1.0

cache = {}
cache["A0"] = X

for i in range(1, L + 1):
    cache["W" + str(i)] = np.random.randn(n[i], n[i-1]) * np.sqrt(2 / 3)
    cache["b" + str(i)] = np.zeros((n[i], 1))


# STEP 4: TRAINING
for iters in range(num_iterations):
    # Forward pass
    for l in range(1, L + 1):
        cache["Z" + str(l)] = np.dot(cache["W" + str(l)], cache["A" + str(l-1)]) + cache["b" + str(l)]
        cache["A" + str(l)] = g[l](cache["Z" + str(l)])

    # Backward pass
    cache["dA" + str(L)] = Loss(cache["A" + str(L)], Y, m, derivative=True)
    for l in range(L, 0, -1):
        cache["dZ" + str(l)] = cache["dA" + str(l)] * g[l](cache["Z" + str(l)], derivative=True)
        cache["dW" + str(l)] = (1/m) * np.dot(cache["dZ" + str(l)], cache["A" + str(l-1)].T)
        cache["db" + str(l)] = (1/m) * np.sum(cache["dZ" + str(l)], axis=1, keepdims=True)
        if l > 1:
            cache["dA" + str(l-1)] = np.dot(cache["W" + str(l)].T, cache["dZ" + str(l)])

    # Update parameters
    for l in range(1, L + 1):
        cache["W" + str(l)] -= cache["dW" + str(l)] * learning_rate
        cache["b" + str(l)] -= cache["db" + str(l)] * learning_rate

    if iters % loss_notification == 0:
        print(f"Iteration {iters}, Loss: {Loss(cache['A' + str(L)], Y, m)}")
    

# STEP 5: TEST
m_test = test_images.shape[0]
X_test = test_images.reshape(m_test, n[0]).T / 255
Y_test = np.zeros((10, m_test))
for i, label in enumerate(test_labels):
    Y_test[label, i] = 1.0
    
test_cache = {}
test_cache["A0"] = X_test

for l in range(1, L + 1):
    test_cache["Z" + str(l)] = np.dot(cache["W" + str(l)], test_cache["A" + str(l-1)]) + cache["b" + str(l)]
    test_cache["A" + str(l)] = g[l](test_cache["Z" + str(l)])

print(f"Test Loss: {Loss(test_cache['A' + str(L)], Y_test, m_test)}")
    
predictions = np.argmax(test_cache['A' + str(L)], axis=0)
labels = np.argmax(Y_test, axis=0)
accuracy = np.mean(predictions == labels)
print(f"Test Accuracy: {accuracy*100:.2f}%")
