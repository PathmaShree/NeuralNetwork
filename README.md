# MNIST Neural Network from Scratch (NumPy)

A 2-layer neural network that recognizes handwritten digits (0-9), built using only NumPy. No TensorFlow or PyTorch. Forward propagation, backpropagation, and gradient descent are all implemented by hand.

## Results

| Metric | Accuracy |
|---|---|
| Training (500 iterations) | ~85-88% |
| Dev set (1,000 unseen images) | ~85-88% |

Results vary slightly between runs because of random shuffling and weight initialization.

## Architecture

```
Input (784)  ->  Hidden (10, ReLU)  ->  Output (10, Softmax)
```

- **Input:** 28x28 grayscale image flattened to 784 pixels, scaled to 0-1
- **Hidden layer:** 10 units with ReLU activation
- **Output layer:** 10 units (one per digit) with softmax activation

### Forward pass

```
Z1 = W1 . X + b1
A1 = ReLU(Z1)
Z2 = W2 . A1 + b2
A2 = softmax(Z2)
```

### Backward pass

```
dZ2 = A2 - one_hot(Y)
dW2 = (1/m) dZ2 . A1^T
db2 = (1/m) sum(dZ2)
dZ1 = W2^T . dZ2 * ReLU'(Z1)
dW1 = (1/m) dZ1 . X^T
db1 = (1/m) sum(dZ1)
```

Parameters are updated with `param = param - alpha * gradient`.

## Requirements

- Python 3.8+
- NumPy
- Matplotlib

```bash
pip install numpy matplotlib
```

## Dataset

Download the MNIST files and place them in a folder named `MNIST` next to the script:

```
project/
  NeuralNetwork.py
  MNIST/
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
```

The files are read directly in their native IDX format, so no extra libraries are needed.

## Usage

```bash
python -m NeuralNetwork
```

The script will:

1. Load the MNIST images and labels and show the first image.
2. Shuffle the data and split it into a 1,000-image dev set and a 59,000-image training set.
3. Train for 500 iterations, printing accuracy every 10 iterations.
4. Print the dev set accuracy.
5. Show four sample predictions. Each image window displays its index, true label, and the network's prediction. Close a window to see the next one.

## Configuration

Edit these in `NeuralNetwork.py`:

| Setting | Location | Default |
|---|---|---|
| Iterations | `gradient_descent(..., iterations=500, ...)` | 500 |
| Learning rate | `gradient_descent(..., alpha=0.1)` | 0.1 |
| Hidden units | shapes in `init_params()` | 10 |
| Dev set size | slicing in the data split section | 1,000 |

## Project structure

| Function | Purpose |
|---|---|
| `load_mnist_images` / `load_mnist_labels` | Read the raw IDX files |
| `init_params` | Random weights and biases in [-0.5, 0.5] |
| `ReLU`, `softmax` | Activation functions (softmax is numerically stable) |
| `forward_prop` | Compute activations layer by layer |
| `one_hot` | Convert labels to one-hot vectors |
| `back_prop` | Compute gradients |
| `update_params` | Gradient descent step |
| `get_predictions`, `get_accuracy` | Evaluation helpers |
| `gradient_descent` | Training loop |
| `make_predictions`, `test_prediction` | Predict and visualize individual images |

## Tips for improving accuracy

- **Train longer:** 1,000-2,000 iterations, since accuracy is still climbing at 500.
- **Increase the learning rate:** try 0.3-0.5.
- **Widen the hidden layer:** 64-128 units typically reaches the mid-90s. Change `W1` to `(128, 784)`, `b1` to `(128, 1)`, and `W2` to `(10, 128)` in `init_params`.
- **Use mini-batches:** updating on 64-128 samples at a time converges faster than full-batch descent.

## Common issues

- **Accuracy stuck near 10%:** weights were initialized with `randn - 0.5`, which biases them negative and kills the ReLU units. Use `rand - 0.5`.
- **Very slow learning:** a learning rate of 0.01 is too low for full-batch gradient descent. Use 0.1 or higher.
- **Overflow warnings in softmax:** subtract the column max before `exp` (already done in this code).
