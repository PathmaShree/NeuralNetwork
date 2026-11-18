import matplotlib.pyplot as plt
import numpy as np


def load_mnist_images(filename):
  with open(filename, "rb") as f:
    magic, num, rows, cols = np.frombuffer(f.read(16), dtype=">i4", count=4)
    return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)


def load_mnist_labels(filename):
  with open(filename, "rb") as f:
    magic, num = np.frombuffer(f.read(8), dtype=">i4", count=2)
    return np.frombuffer(f.read(), dtype=np.uint8)


image_file = "MNIST/train-images-idx3-ubyte"
label_file = "MNIST/train-labels-idx1-ubyte"

images = load_mnist_images(image_file)
labels = load_mnist_labels(label_file)

# Verify
print("Label:", labels[0])
plt.imshow(images[0], cmap="gray")
plt.show()
print("First 5 labels:", labels[:5])

# 1. Flatten images to (60000, 784)
X = images.reshape(images.shape[0], -1)
Y = labels

# 2. Shuffle images and labels together
combined = np.column_stack((Y, X))
np.random.shuffle(combined)

Y_shuffled = combined[:, 0].astype(int)   # FIX: cast labels to int
X_shuffled = combined[:, 1:]

# 3. Dev set (first 1000), transposed -> (784, 1000)
X_dev = X_shuffled[0:1000].T / 255.0
Y_dev = Y_shuffled[0:1000]

# 4. Train set (rest), transposed -> (784, 59000)
X_train = X_shuffled[1000:].T / 255.0
Y_train = Y_shuffled[1000:]

print("--- Data Shapes ---")
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("Done processing! Ready to build the network.")


def init_params():
  # FIX: rand (uniform 0-1) minus 0.5 gives weights in [-0.5, 0.5]
  W1 = np.random.rand(10, 784) - 0.5
  b1 = np.random.rand(10, 1) - 0.5
  W2 = np.random.rand(10, 10) - 0.5
  b2 = np.random.rand(10, 1) - 0.5
  return W1, b1, W2, b2


def ReLU(Z):
  return np.maximum(0, Z)


def softmax(Z):
  # FIX: subtract max for numerical stability
  Z = Z - np.max(Z, axis=0, keepdims=True)
  expZ = np.exp(Z)
  return expZ / np.sum(expZ, axis=0, keepdims=True)


def forward_prop(W1, b1, W2, b2, X):
  Z1 = W1.dot(X) + b1
  A1 = ReLU(Z1)
  Z2 = W2.dot(A1) + b2
  A2 = softmax(Z2)
  return Z1, A1, Z2, A2


def one_hot(Y):
  one_hot_Y = np.zeros((Y.size, Y.max() + 1))
  one_hot_Y[np.arange(Y.size), Y] = 1
  return one_hot_Y.T


def deriv_ReLU(Z):
  return Z > 0


def back_prop(Z1, A1, Z2, A2, W2, X, Y):
  m = Y.size
  one_hot_Y = one_hot(Y)
  dZ2 = A2 - one_hot_Y
  dW2 = 1 / m * dZ2.dot(A1.T)
  db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
  dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
  dW1 = 1 / m * dZ1.dot(X.T)
  db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
  return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
  W1 = W1 - alpha * dW1
  b1 = b1 - alpha * db1
  W2 = W2 - alpha * dW2
  b2 = b2 - alpha * db2
  return W1, b1, W2, b2


def get_predictions(A2):
  return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
  # FIX: removed the noisy print of full arrays
  return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, alpha):
  W1, b1, W2, b2 = init_params()
  for i in range(iterations):
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
    dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
    W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    if i % 10 == 0:
      print("Iteration:", i, "| Accuracy:", get_accuracy(get_predictions(A2), Y))
  return W1, b1, W2, b2


def make_predictions(X, W1, b1, W2, b2):
  _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
  predictions = get_predictions(A2)
  return predictions


def test_prediction(index, W1, b1, W2, b2):
  current_image = X_train[:, index, None]   # keep shape (784, 1)
  prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
  label = Y_train[index]
  print("Prediction: ", prediction)
  print("Label: ", label)

  current_image = current_image.reshape((28, 28)) * 255
  plt.gray()
  plt.imshow(current_image, interpolation="nearest")
  plt.show()


if __name__ == "__main__":
  # FIX: higher learning rate (0.01 was far too slow)
  W1, b1, W2, b2 = gradient_descent(X_train, Y_train, iterations=500, alpha=0.1)

  # Check generalization on the dev set
  _, _, _, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
  print("Dev accuracy:", get_accuracy(get_predictions(A2_dev), Y_dev))


  # Visualize a few predictions (close each image window to see the next)
  test_prediction(0, W1, b1, W2, b2)
  test_prediction(1, W1, b1, W2, b2)
  test_prediction(2, W1, b1, W2, b2)
  test_prediction(3, W1, b1, W2, b2)