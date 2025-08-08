"""Define functions to sample from the von Mises-Fisher distribution."""

import numpy as np
from numpy.typing import NDArray


def sample_uniform_hypersphere(n: int, num_samples: int, rng: np.random.Generator) -> NDArray:
    """Sample uniform positions on an n-dimensional unit sphere.

    Note: Here, "n-sphere" is used in the topological sense, meaning a sphere with a surface
        dimension of n. In this terminology, the "usual sphere" is called the 2-sphere.

    Reference:
        Sphere (Wolfram MathWorld).
        https://mathworld.wolfram.com/Sphere.html

        Sphere Point Picking (Wolfram MathWorld).
        https://mathworld.wolfram.com/SpherePointPicking.html

    :param n: Topological dimension of the hypersphere on which samples are generated
    :param num_samples: Number of samples to be generated
    :return: NumPy array of uniform (n + 1)-dim. samples on the unit n-sphere (# samples, n + 1)
    """
    sample_dim = n + 1  # Generate (n + 1)-dimensional samples to produce an n-sphere
    samples = rng.normal(size=(num_samples, sample_dim))
    sample_norms = np.linalg.norm(samples, axis=1, keepdims=True)  # Shape (num_samples, 1)
    return np.divide(samples, sample_norms)
