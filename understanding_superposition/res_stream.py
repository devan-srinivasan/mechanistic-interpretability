import numpy as np
from basis import tao_construction, projection_matrix, test_complete_combined_error
from vis import plot_heatmap

dim_in = 50
n = 225
sbasis = tao_construction(0, dim_in, width=n)
dim = sbasis.shape[1]

def visualize_orthogonality():
    S = sbasis @ sbasis.T - np.eye(sbasis.shape[0])
    e = 0.005
    # G = (np.abs(S) < e) * np.ones(sbasis.shape[0])
    plot_heatmap([S])

def project(basis_vec: np.ndarray, vec: np.ndarray) -> np.ndarray:
    mat = projection_matrix(basis_vec)
    return mat @ vec

visualize_orthogonality()

# stream = np.zeros((dim,1))