import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_fn
from basis import tao_construction
import seaborn as sns
import matplotlib.pyplot as plt

B = tao_construction(n_vecs=(766/2) ** 2, dim=766, width=6 ** 2)
# B = tao_construction(n_vecs=7**2, dim=14, width=4**2)

# cos = cos_fn(B, B) # shape (n_vecs, n_vecs)

# low_tri = cos[np.tril_indices(B.shape[0], k=-1)]

# print(f"{np.abs(low_tri).mean()} {np.abs(low_tri).std()} {np.abs(low_tri).max()} {np.abs(low_tri).min()}")

alphabet = ['a', 'b', 'c', '[eos]']

in_use_mask = np.zeros(B.shape[0], dtype=bool)

pos_space = B[:2]
alphabet_space = B[6:6 + len(alphabet)]
next_alphabet_space = B[12:12 + len(alphabet)]
output_space = B[18:18 + len(alphabet) - 1]

in_use_mask[6:6 + len(alphabet)] = True
in_use_mask[0:2] = True

seq = 'a b a c a [eos]'

streams = np.zeros((len(seq.split()), B.shape[1]))

def print_mask(in_use_mask):
    for i in range(0, len(in_use_mask), 6):
        print(in_use_mask[i:i+6])

def read_fn(S: np.ndarray):
    # S: (N, d), rows are vectors spanning the subspace

    # Compute (S S^T)^{-1}
    M = S @ S.T
    M_inv = np.linalg.pinv(M)  # more stable than inv

    # Projection matrix
    P = S.T @ M_inv @ S
    return P

def rotation_fn(S, theta):
    # using a composition of two Householder reflections we can create a dxd 
    # rotation matrix that rotates a vector in the plane spanned by pos1 and pos2 by an angle theta

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

# Feature Embedding
# positional embedding
theta = 2 * np.pi / (streams.shape[0] + 1)
angles = np.linspace(0, 2 * np.pi, (streams.shape[0] + 1), endpoint=False)
for i, theta_ in enumerate(angles[:streams.shape[0]]):
    R = rotation_fn(pos_space, theta_)
    streams[i] = R @ pos_space[0]
    
# token embedding
for i, token in enumerate(seq.split()):
    idx = alphabet.index(token)
    streams[i] += alphabet_space[idx]

# Layer 1

W_Q = 4 * rotation_fn(pos_space, theta) @ read_fn(pos_space)
W_K = 4 * read_fn(pos_space)

A = streams @ W_Q.T @ W_K @ streams.T

# print("Layer 1: copy next token to previous token")
# print(np.array2string(softmax(A, axis=-1), formatter={'float_kind':lambda x: f"{x:.2f}"}))

W_V = read_fn(alphabet_space)
W_O = translation_fn(alphabet_space, next_alphabet_space)

streams += (softmax(A, axis=-1) @ (streams @ W_V.T)) @ W_O.T

# Layer 2

W_Q = 2 * read_fn(alphabet_space)
W_K = W_Q

A = streams @ W_Q.T @ W_K @ streams.T

# print("Layer 2: copy between matching tokens")
# print(np.array2string(softmax(A, axis=-1), formatter={'float_kind':lambda x: f"{x:.2f}"}))

W_V = 3 * translation_fn_with_kernel(next_alphabet_space, output_space, kernel=next_alphabet_space[-1].reshape(1,-1)) @ read_fn(next_alphabet_space)

# print()
# print(np.array2string(streams @ W_V.T @ alphabet_space.T, formatter={'float_kind':lambda x: f"{x:.2f}"}))

# we want to send [mask] vector in next_space to 0
W_O = np.eye(streams.shape[1])

streams += (softmax(A, axis=-1) @ (streams @ W_V.T)) @ W_O.T

# MLM head

W_head = output_space @ read_fn(output_space)

print("input:", seq.split()[:-1])
print(f"next: {alphabet[np.argmax(streams[-2] @ W_head.T)]}")