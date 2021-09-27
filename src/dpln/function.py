import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def softmax(z: np.ndarray, axis=-1) -> np.ndarray:
    """
    softmax

    :param z: shape(batch_size, in)
    :param axis: -1 for default
    :return: shape(batch_size, in)
    """
    return np.exp(z) / np.sum(np.exp(z), axis=axis, keepdims=True)


# def softmax_derivative(z: np.ndarray, axis=-1) -> np.ndarray:
#     """
#     softmax derivative

#     :param z: shape(batch_size, in)
#     :param axis: -1
#     :return: shape(batch_size, in, in)
#     """
#     x = softmax(z, axis=axis)  # (bs, 10)
#     s = x[:, :, np.newaxis]  # (bs, 10, 1)
#     a = np.array([np.diagflat(s[i, :, :]) for i in range(s.shape[0])])  # (bs, 10, 10)
#     b = np.matmul(s, s.transpose((0, 2, 1)))  # (bs, 10, 10)
#     return a - b

# def softmax_derivative(Q):
#     x=softmax(Q)
#     s=x.reshape(-1,1)
#     return (np.diagflat(s) - np.dot(s, s.T))
