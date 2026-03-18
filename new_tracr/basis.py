import numpy as np
from tqdm import tqdm

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

