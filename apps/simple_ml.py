"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname, 'rb') as image_file:
        data_magic, data_nums, data_rows, data_cols = struct.unpack('>IIII', image_file.read(16))
        print(f"image data meta info: Magic:{data_magic}, Nums:{data_nums}, Rows:{data_rows}, Cols:{data_cols}")
        data_buffer = image_file.read(data_nums * data_rows * data_cols)
        data = np.frombuffer(data_buffer, dtype=np.uint8)
        data = np.reshape(data, (data_nums, data_rows * data_cols)).astype(np.float32)
        data = data / 255.0
    with (gzip.open(label_filename, 'rb')) as label_file:
        label_magic, label_nums = struct.unpack('>II', label_file.read(8))
        print(f"image label meta info: Magic:{label_magic}, Nums:{label_nums}")
        label_buffer = label_file.read(label_nums)
        label = np.frombuffer(label_buffer, dtype=np.uint8)
    return data, label
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot) -> ndl.Tensor:
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    from needle.ops import log, summation, exp, add, multiply, divide_scalar
    batch_size = Z.shape[0]
    log_of_sum_exp = log(summation(exp(Z), axes=(-1,)))
    cor_prediction = summation(multiply(Z, y_one_hot), axes=(-1,))
    return summation(log_of_sum_exp - cor_prediction) / batch_size
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    ### BEGIN YOUR SOLUTION
    def to_onehot(indexes: np.ndarray, batch_size: int, num_classes: int):
        ret = np.zeros((batch_size, num_classes))
        ret[np.arange(batch_size), indexes] = 1
        return ret
    from needle.ops import matmul, relu
    assert X.shape[0] == y.shape[0]
    num_examples = X.shape[0]
    num_classes = np.max(y) + 1
    for begin_index in range(0, num_examples, batch):
        # calc index
        cur_batch_size = min(batch, num_examples - begin_index)

        # transfer input x
        x_batch = ndl.Tensor(X[begin_index: begin_index + cur_batch_size])
        y_batch = ndl.Tensor(
            to_onehot(y[begin_index: begin_index + cur_batch_size], cur_batch_size, num_classes)
        )

        # forward to z
        z_batch = matmul(relu(matmul(x_batch, W1)), W2)

        # calc batch loss & backward propagation
        batch_loss = softmax_loss(z_batch, y_batch)
        batch_loss.backward()

        # update gradients
        W1.data -= lr * ndl.Tensor(W1.grad.numpy().astype(np.float32))
        W2.data -= lr * ndl.Tensor(W2.grad.numpy().astype(np.float32))
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
