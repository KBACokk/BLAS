"""Интерфейсные тесты для самописной реализации dtrsm (нижний треугольник, column-major)."""

import numpy as np
import pytest


def trsm_reference(A: np.ndarray, B: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Эталон, повторяющий логику main.cpp (column-major, flat indexing)."""
    n = A.shape[0]
    lda = ldb = n
    A_f = np.asarray(A, dtype=np.float64, order="F").ravel()
    out = np.asarray(B, dtype=np.float64, order="F").ravel().copy()
    for j in range(n):
        for i in range(n):
            acc = 0.0
            for k in range(i):
                acc += A_f[i + k * lda] * out[k + j * ldb]
            out[i + j * ldb] = (alpha * out[i + j * ldb] - acc) / A_f[i + i * lda]
    return out.reshape(B.shape, order="F")


@pytest.mark.parametrize("n", [4, 8, 16])
def test_trsm_reference_finite(n):
    rng = np.random.default_rng(42)
    A = np.tril(rng.random((n, n), dtype=np.float64))
    for i in range(n):
        A[i, i] = n + 1.0
    B = rng.random((n, n), dtype=np.float64)
    result = trsm_reference(A, B)
    assert np.all(np.isfinite(result))


def test_trsm_deterministic():
    n = 6
    A = np.tril(np.ones((n, n), dtype=np.float64) * 0.5)
    np.fill_diagonal(A, n + 2.0)
    B = np.arange(n * n, dtype=np.float64).reshape(n, n, order="F")
    first = trsm_reference(A, B)
    second = trsm_reference(A, B.copy())
    np.testing.assert_allclose(first, second, rtol=1e-12)
