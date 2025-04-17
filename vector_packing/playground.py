import numpy as np
from communities.algorithms import bron_kerbosch
from communities.visualization import draw_communities
from fastJLT import FastJLT
from tqdm import tqdm

def project_basis_vectors(original_dim: int, target_dim: int):
    basis_vectors = np.eye(original_dim)
    fjlt_transformer = FastJLT(target_dim=target_dim)
    projected_vectors = fjlt_transformer.fit_transform(basis_vectors)

    return projected_vectors.T

def cosine_similarity_matrix(X):
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X_normalized @ X_normalized.T

def cosine_similarity_elementwise(A, B):
    # Compute dot products for each row pair: shape (n,)
    dot_product = np.sum(A * B, axis=1)

    # Compute norms for each row: shape (n,)
    norm_A = np.linalg.norm(A, axis=1)
    norm_B = np.linalg.norm(B, axis=1)

    # Avoid division by zero
    denom = norm_A * norm_B
    denom[denom == 0] = 1e-10  # or handle however you prefer
    
    return dot_product / denom

def construct_graph(vectors: np.ndarray, epsilon: float = 1.0):
    sims = cosine_similarity_matrix(vectors) - np.eye(vectors.shape[0])
    adj = (np.abs(sims) < epsilon) * np.ones(sims.shape)
    return adj
         
def greedy_maximal_clique_from_matrix(G):
    # Get the number of nodes in the graph
    n = G.shape[0]
    
    # Initialize an empty list to store the clique
    clique = []
    
    # Create a list of nodes sorted by degree (highest degree first)
    degrees = np.sum(G, axis=1)  # Sum across rows to get the degree of each node
    nodes = np.argsort(degrees)[::-1]  # Sort nodes by degree in descending order
    
    # Iterate through the nodes and try to add them to the clique
    for node in nodes:
        # Check if the node is connected to all nodes in the current clique
        if all(G[node, other] == 1 for other in clique):
            clique.append(node)
    
    return clique

def greedy_maximal_clique_from_matrix_devan(G):
    # Get the number of nodes
    n = G.shape[0]
    
    # Compute degrees
    degrees = G.sum(axis=1)

    # Start with the node with the highest degree
    current = np.argmax(degrees)
    clique = [current]
    
    # Candidates are those connected to the current node
    candidates = set(np.where(G[current])[0])
    
    while candidates:
        # For each candidate, count how many connections it has to the current clique
        overlaps = []
        for node in candidates:
            overlap = sum(G[node, c] for c in clique)
            overlaps.append((overlap, degrees[node], node))
        
        # Pick the node with the most overlaps (break ties with degree)
        overlaps.sort(reverse=True)
        _, _, best = overlaps[0]
        
        # Check if it's connected to all nodes in current clique
        if all(G[best, c] for c in clique):
            clique.append(best)
            # Update candidates to those connected to all nodes in clique
            candidates = {i for i in candidates if i != best and all(G[i, c] for c in clique)}
        else:
            break

    return clique

def matthew_daws_construction(n: int, expl: int = 500):
    # from: https://mathoverflow.net/questions/24864/almost-orthogonal-vectors
    sbasis = np.eye(n)
    
    def mutually_orthogonal(vectors: np.ndarray, new: np.ndarray, interference: float = 0.1):
        # we assume vectors are already mutually orthogonal, and normalized
        return bool(np.dot(vectors, new).max() < interference)
    
    level = 0
    vec = np.ones((1, n)) / np.sqrt(n)
    while 2**level < n and mutually_orthogonal(sbasis, vec.T):
        assert(vec.shape[1] == n)
        sbasis = np.concatenate((sbasis, vec), axis=0)
        level += 1

        vec = np.tile(
            np.concatenate((
                np.ones((1, n // 2**level)), 
                np.ones((1, n // 2**level)) * -1
            ), axis=1),
            2 ** (level - 1)
        )
        # normalize
        vec /= np.sqrt(n)

    for _ in (pbar := tqdm(range(expl * n), unit='samples', leave=True)):
        vec = np.random.choice([-1.0, 1.0], size=n).reshape(1,n) / np.sqrt(n)
        if mutually_orthogonal(sbasis, vec.T):
            sbasis = np.concatenate((sbasis, vec), axis=0)
        # pbar.update(1)
        pbar.set_postfix({'new': sbasis.shape[0] - n})
    pbar.close()
    return sbasis
 
def projection_matrix(u: np.ndarray) -> np.ndarray:
    assert(u.shape[0] == 1 or len(u.shape) == 1) # row vector
    u = u.reshape(-1, 1)  # convert to column vector
    return (u @ u.T) / (u.T @ u)

def generate_quadratic_phase_matrix(p):
    # All values of x in F_p
    x = np.arange(p)[None, :]  # shape (1, p)

    # All possible values for a, b in F_p
    a = np.arange(p)
    b = np.arange(p)

    # Create grid of all (a, b) pairs
    a_grid, b_grid = np.meshgrid(a, b, indexing='ij')  # shape (p, p)
    a_flat = a_grid.reshape(-1, 1)  # shape (p^2, 1)
    b_flat = b_grid.reshape(-1, 1)  # shape (p^2, 1)

    # Compute ax^2
    ax2 = (a_flat * (x ** 2)) % p  # shape (p^2, p)

    # Compute bx
    bx = (b_flat * x) % p  # shape (p^2, p)

    # Compute a x^2 + b x
    exponent = (ax2 + bx) % p  # shape (p^2, p)

    # Compute complex exponential e^{2πi (ax^2 + bx)/p}
    angles = 2 * np.pi * exponent / p
    complex_vals = np.exp(1j * angles) / np.sqrt(p)  # shape (p^2, p)

    # Convert to real-valued 2p vectors: [Re, Im]
    V_real = complex_vals.real  # shape (p^2, p)
    V_imag = complex_vals.imag  # shape (p^2, p)
    V = np.concatenate([V_real, V_imag], axis=1)  # shape (p^2, 2p)

    return V

def is_prime(n):
    """Deterministic Miller-Rabin primality test for n < 2^64."""
    if n < 2:
        return False
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def next_prime(n):
    """Find the smallest prime greater than n."""
    if n < 2:
        return 2
    candidate = n + 1 if n % 2 == 0 else n + 2
    while not is_prime(candidate):
        candidate += 2
    return candidate

def tao_construction(dim: int):
    p = next_prime(dim)
    sbasis = generate_quadratic_phase_matrix(p)
    return sbasis

def test_error(sbasis: np.ndarray):
    S = sbasis @ sbasis.T - np.eye(sbasis.shape[0])
    e = 0.05
    G = (np.abs(S) < e) * np.ones(sbasis.shape[0])

    n_components = sbasis.shape[0]
    # indices = np.random.randint(dim, sbasis.shape[0], (n_components,))
    indices = np.random.choice(sbasis.shape[0], size=(n_components,), replace=False)   # must work since they are completely orthogonal

    vector_components = sbasis[indices]
    # scalars = np.ones((n_components,1)) * 50
    scalars = np.random.randint(1, 1000, (n_components,1))
    combined_vector = (vector_components * scalars).sum(axis=0, keepdims=True)

    projection_matrices = np.array([
        projection_matrix(vector_components[i]) for i in range(n_components)
    ])

    # reconstructed_components = np.einsum('ijk,bk->ij', projection_matrices, combined_vector)
    reconstructed_components = np.matmul(projection_matrices, combined_vector.squeeze(0))
    elementwise_error = np.abs(reconstructed_components - vector_components * scalars)
    cosine_error = cosine_similarity_elementwise(reconstructed_components, vector_components * scalars)
    norm_error = np.abs(np.linalg.norm(reconstructed_components, axis=1) - np.linalg.norm(vector_components * scalars, axis=1))

    print(f"""
Element-wise Error: 
    Max: {elementwise_error.max().item():.4f}
    Mean: {elementwise_error.mean().item():.4f}

Cosine Error: 
    Max: {1 - cosine_error.max().item():.5f}
    Mean: {1 - cosine_error.mean().item():.5f}

Norm Error: 
    Max: {norm_error.max().item():.4f}
    Mean: {norm_error.mean().item():.4f}
    """)