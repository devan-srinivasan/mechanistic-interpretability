import numpy as np
from basis import tao_construction, projection_matrix, test_complete_combined_error
from vis import plot_matrix

b = tao_construction(n_vecs=None, dim=760, width=784)
dim = b.shape[1]

def visualize_orthogonality():
    S = b @ b.T - np.eye(b.shape[0])
    e = 0.005
    # G = (np.abs(S) < e) * np.ones(b.shape[0])
    plot_matrix(S)

def project(basis_vec: np.ndarray, vec: np.ndarray) -> np.ndarray:
    mat = projection_matrix(basis_vec)
    return mat @ vec

visualize_orthogonality()

# stream = np.zeros((dim,1))