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

def generate_quadratic_phase_matrix(p, width=None):
    x = np.arange(p)[None, :]  # shape (1, p)

    if width is None:
        # Full F_p × F_p grid
        a = np.arange(p)
        b = np.arange(p)
    else:
        k = int(np.sqrt(width))
        assert k * k == width, "width must be a perfect square"
        assert k <= p, "√width must be ≤ p"
        a = np.arange(k)
        b = np.arange(k)

    # Create grid of (a, b) pairs
    a_grid, b_grid = np.meshgrid(a, b, indexing='ij')  # shape (k, k) or (p, p)
    a_flat = a_grid.reshape(-1, 1)  # shape (width, 1)
    b_flat = b_grid.reshape(-1, 1)  # shape (width, 1)

    # Compute ax^2 + bx mod p
    exponent = (a_flat * x**2 + b_flat * x) % p  # shape (width, p)
    angles = 2 * np.pi * exponent / p
    complex_vals = np.exp(1j * angles) / np.sqrt(p)

    # Convert to real-valued 2p vectors
    V = np.concatenate([complex_vals.real, complex_vals.imag], axis=1)  # shape (width, 2p)

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
    candidate = n + 1 if n % 2 == 0 else n
    while not is_prime(candidate):
        candidate += 2
    return candidate

def tao_construction(n_vecs: int, dim: int = None, width: int = None) -> np.ndarray:
    """
    will return atleast p^2 semi-orthogonal 2p dimensional vectors V with pairwise inner products of 
    at most 1/root(p). p^2 >= n_vecs is ensured.
    """
    if dim is None:
        p = next_prime(int(np.ceil(np.sqrt(n_vecs))))
    else:
        p = next_prime(int(dim // 2) + dim % 2)
    # print(p)
    sbasis = generate_quadratic_phase_matrix(p, width=width)
    return sbasis

def random_construction(n_vecs: int, dim: int, sample: int = 5000, k: int = 10):
    vecs = np.random.randn(sample, dim)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    S = np.abs(vecs @ vecs.T)
    # beam search now
    best_idxs = beam_search(S, n_vecs, k)
    assert(best_idxs.shape[0] == n_vecs)
    return vecs[best_idxs]

def beam_search(S: np.ndarray, n: int, k: int):
    num_nodes = S.shape[0]

    # Step 1: Initialize beams with lowest average similarities
    initial_scores = S.mean(axis=1)
    beams = np.argsort(initial_scores)[:k].reshape(k, 1)  # shape (k, 1)

    for _ in tqdm(range(n - 1)):
        # Step 2: For each beam, we want to expand it with k new completions

        # Flatten beams so we can vectorize
        num_beams = beams.shape[0]

        # Build a (num_beams, num_nodes) boolean mask where True = candidate
        candidate_mask = np.ones((num_beams, num_nodes), dtype=bool)
        for i, beam in enumerate(beams):
            candidate_mask[i, beam] = False  # mask out existing nodes

        # Compute score: for each beam, get the *max* similarity between current cluster and all others
        # shape of S[beam][:, candidate]: (len(beam), num_nodes), max over axis=0 gives (num_nodes,)
        max_scores = np.full((num_beams, num_nodes), np.inf)
        for i, beam in enumerate(beams):
            max_scores[i] = S[beam][:, :].max(axis=0)  # max sim between beam and each candidate

        # Mask out ineligible candidates (already in beam)
        max_scores[~candidate_mask] = np.inf

        # Now for each beam, pick the k lowest-scoring nodes (i.e., low similarity)
        topk_indices = np.argpartition(max_scores, kth=k, axis=1)[:, :k]  # shape (num_beams, k)

        # Gather new candidate beams (shape: num_beams * k, beam_len + 1)
        # Repeat each beam `k` times
        repeated_beams = np.repeat(beams, k, axis=0)  # shape (k^2, beam_len)

        # Flatten top-k new nodes to append
        new_nodes = topk_indices.flatten()            # shape (k^2,)

        # Concatenate to form new beams
        new_beams = np.concatenate(
            [repeated_beams, new_nodes[:, None]],
            axis=1
        )  # shape (k^2, beam_len + 1)

        # Get scores for each new node
        beam_indices = np.repeat(np.arange(k), k)     # shape (k^2,)
        new_scores = max_scores[beam_indices, new_nodes]  # shape (k^2,)

        # Pick top k by score
        topk = np.argsort(new_scores)[:k]
        beams = new_beams[topk]

    return beams[0]

def test_complete_combined_error(sbasis: np.ndarray, indices: np.ndarray = None, n_components: int = 5):
    S = sbasis @ sbasis.T - np.eye(sbasis.shape[0])
    e = 0.05
    # G = (np.abs(S) < e) * np.ones(sbasis.shape[0])

    # indices = np.random.randint(dim, sbasis.shape[0], (n_components,))
    if indices is None:
        indices = np.random.choice(sbasis.shape[0], size=(n_components,), replace=False)   # must work since they are completely orthogonal
    else:
        n_components = indices.shape[0]
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