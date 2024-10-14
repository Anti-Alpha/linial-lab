import numpy as np

def get_matrix(n: int, m: int) -> np.ndarray:
    """Create random matrix n * m."""
    return np.random.rand(n, m)

def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrix addition."""
    return np.add(x, y)

def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Matrix multiplication by scalar."""
    return np.multiply(x, a)

def dot_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrices dot product."""
    return np.dot(x, y)

def identity_matrix(dim: int) -> np.ndarray:
    """Create identity matrix with dimension `dim`."""
    return np.eye(dim)

def matrix_inverse(x: np.ndarray) -> np.ndarray:
    """Compute inverse matrix."""
    return np.linalg.inv(x)

def matrix_transpose(x: np.ndarray) -> np.ndarray:
    """Compute transpose matrix."""
    return np.transpose(x)

def hadamard_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Hadamard product."""
    return np.multiply(x, y)

def basis(x: np.ndarray) -> tuple[int]:
    """Compute matrix basis (column pivots)."""
    _, _, p = np.linalg.svd(x, full_matrices=False)
    return tuple(p)

def norm(x: np.ndarray, order: int | float | str) -> float:
    """Matrix norm: Frobenius, Spectral or Max."""
    return np.linalg.norm(x, ord=order)
