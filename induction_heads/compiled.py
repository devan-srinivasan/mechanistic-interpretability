import numpy as np
# input
n_vocab = 4
seq_len = 3
d = 3

# config
vocab = list(range(1, n_vocab + 1)) # [1...n_vocab]

def projection_matrix(u: np.ndarray) -> np.ndarray:
    assert(u.shape[0] == 1 or len(u.shape) == 1) # row vector
    u = u.reshape(-1, 1)  # convert to column vector
    return (u @ u.T) / (u.T @ u)

W_QK = np.zeros((seq_len, seq_len))
W_OV = projection_matrix(np.array([0, 0, 1]).reshape((1, d)))