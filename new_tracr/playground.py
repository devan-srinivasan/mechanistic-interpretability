import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_fn
from basis import tao_construction
import seaborn as sns
import matplotlib.pyplot as plt

B = tao_construction(n_vecs=(766/2) ** 2, dim=766, width=50 ** 2)

class Subspace:
    def __init__(self, vectors, basis_names: list[str]):
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)  # shape (1, d)

        self.S = vectors  # shape (k, d)
        self.dim = self.S.shape[0]
        self.labels = basis_names

    def proj(self):
        # S: (N, d), rows are vectors spanning the subspace
        # Compute (S S^T)^{-1}
        M = self.S @ self.S.T
        M_inv = np.linalg.pinv(M)  # more stable than inv

        # Projection matrix
        P = self.S.T @ M_inv @ self.S
        return P
    
    def debug(self, X: np.ndarray):
        """
        Computes "presence" of subspace axis in each vector in X, useful for debugging.
        X is shape (..., d)
        """
        A = np.dot(X, self.S.T)  # shape (m, k)
        A = softmax(A, axis=1)
        present = []
        for i in range(A.shape[0]):
            sig_indices = np.where(A[i] > 0.4)[0]
            present.append([self.labels[idx] for idx in sig_indices])
        return present

def rotation_fn(S, theta):
    # using a composition of two Householder reflections we can create a dxd 
    # rotation matrix that rotates a vector in the plane spanned by pos1 and pos2 by an angle theta

    assert S.shape[0] == 2
    pos1, pos2 = S[0], S[1]

    def householder(u):
        u = u / np.linalg.norm(u)
        return np.eye(len(u)) - 2 * np.outer(u, u)

    # Orthonormalize pos1, pos2 → u, v
    u = pos1 / np.linalg.norm(pos1)
    v = pos2 - np.dot(pos2, u) * u
    v = v / np.linalg.norm(v)

    # Construct intermediate vector at half-angle
    w = np.cos(theta / 2) * u + np.sin(theta / 2) * v

    # Two Householder reflections
    H1 = householder(u)
    H2 = householder(w)

    # Rotation matrix
    R = H2 @ H1
    return R

def translation_fn(S, S_):
    """
    returns linear transformation matrix that translates subspace S to S_
    so S[0] -> S_[0], S[1] -> S_[1], etc.
    """
    # S, S_: (N, d)

    # Convert to column-basis form
    V = S.T      # (d, N)
    V_ = S_.T    # (d, N)

    # Compute pseudoinverse
    V_pinv = np.linalg.pinv(V)

    # Linear map
    T = V_ @ V_pinv
    return T

def translation_fn_with_kernel(S, S_, kernel):
    """
    S: (n, d)
    S_: (n-k, d)
    kernel: (k, d)

    Maps:
      - vectors in kernel → 0
      - remaining vectors in S → corresponding vectors in S_
    Returns: (d, d) matrix
    """

    # Convert to column form
    V = S.T          # (d, n)
    V_kernel = kernel.T  # (d, k)

    # Identify which rows of S are NOT in kernel
    # (assumes exact row matching)
    mask = []
    for row in S:
        if not any(np.allclose(row, krow) for krow in kernel):
            mask.append(True)
        else:
            mask.append(False)
    mask = np.array(mask)

    S_active = S[mask]      # (n-k, d)
    V_active = S_active.T   # (d, n-k)
    V_target = S_.T         # (d, n-k)

    # Map active → S_
    T_active = V_target @ np.linalg.pinv(V_active)

    # Projection onto kernel
    P_kernel = V_kernel @ np.linalg.pinv(V_kernel)

    # Kill kernel, apply active map
    T = T_active @ (np.eye(V.shape[0]) - P_kernel)

    return T

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def RoPE_rotation_fn(m: int, d: int = 766, base: int = 512):
    """
    Returns a (d, d) rotation matrix that applies RoPE-style rotations to the first m dimensions.
    The remaining dimensions are unchanged. m is absolute position
    """
    R = np.eye(d)
    for i in range(0, d, 2):
        theta = m / (base ** (i / d))
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R_i = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        R[i:i+2, i:i+2] = R_i
    return R

X = np.zeros((8, B.shape[1]))

alphabet_space = Subspace(B[:3], basis_names=['a', 'b', 'c'])

ropes = [RoPE_rotation_fn(m, base=512) for m in range(8)]

X += alphabet_space.S[0]  # add 'a' to all vectors

WQ = alphabet_space.proj()
WK = alphabet_space.proj()

Q = X @ WQ
K = X @ WK

rope_Q = np.stack([ropes[i] @ Q[i] for i in range(Q.shape[0])])
rope_K = np.stack([ropes[i] @ K[i] for i in range(K.shape[0])])

A = rope_Q @ rope_K.T
# A = Q @ K.T
# causal_mask = np.tril(np.ones_like(A, dtype=bool))
# A_masked = np.where(causal_mask, A, -np.inf)

sns.heatmap(softmax(5 * A), cmap="Blues", vmin=0, vmax=1)
plt.show()