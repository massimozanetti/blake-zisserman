import numpy as np
import pytest
import scipy.sparse.linalg as sla
from scipy.sparse import spdiags, eye, kron

from blakezisserman import BlakeZisserman1D, BlakeZisserman2D
from blakezisserman import core


def piecewise_signal(rng, n=120):
    x = np.linspace(0, 1, n)
    y = np.where(x < 0.5, x, 1.0 - x)             # gradient discontinuity at x = 0.5
    return y + 0.01 * rng.standard_normal(n)


def test_1d_signal_converges():
    rng = np.random.default_rng(0)
    g = piecewise_signal(rng)
    u, z, t, F, converged = BlakeZisserman1D(values=g).solve(1e-2, 1e-3)
    assert converged
    assert u.shape == (g.size, 1) and z.shape == (g.size,)
    F = F[F != 0]
    assert F[-1] <= F[0]


def test_1d_uneven_spacing_has_no_false_corners():
    # a straight line has no corners, however unevenly it is sampled
    t = np.concatenate([np.arange(0, 20, 0.5), 20 + np.arange(0, 40, 2.0)])   # spacing jumps from 0.5 to 2
    u, z, t_out, F, converged = BlakeZisserman1D(values=0.5 * t, breaks=t).solve(1e-2, 1e-4)
    assert converged
    np.testing.assert_array_equal(t_out, t)
    assert z[1:-1].min() > 0.99            # interior only: the end rows are one-sided


def test_1d_closed_curve():
    # unit circle sampled unevenly; the last sample closes the curve
    t = np.sort(np.r_[0, np.random.default_rng(1).uniform(0, 2 * np.pi, 59), 2 * np.pi])
    g = np.stack([np.cos(t), np.sin(t)], axis=1)
    g[-1] = g[0]
    u, z, t_out, F, converged = BlakeZisserman1D(values=g, breaks=t).solve(1e-1, 1e-1, bound_cond='periodic')
    assert converged
    assert u.shape == g.shape and z.shape == (g.shape[0],)
    np.testing.assert_array_equal(u[-1], u[0])     # output is closed too
    assert z.min() > 0.99                          # smooth circle: no corners


def test_1d_input_checks():
    with pytest.raises(ValueError, match="closed curve"):
        BlakeZisserman1D(values=np.arange(10.)).solve(1e-2, 1e-4, bound_cond='periodic')
    with pytest.raises(ValueError, match="strictly increasing"):
        BlakeZisserman1D(values=np.arange(5.), breaks=np.array([0, 1, 1, 2, 4.]))
    with pytest.raises(ValueError, match="Extra inputs"):
        BlakeZisserman1D(values=np.arange(5.), brakes=np.arange(5.))


@pytest.mark.parametrize("channels", [None, 3])
def test_2d_image_converges(channels):
    rng = np.random.default_rng(2)
    I, J = 24, 30
    img = np.zeros((I, J))
    img[:, J // 2:] = 100.0                         # one sharp edge (0-255 intensity scale)
    if channels:
        img = np.repeat(img[:, :, None], channels, axis=2)
    img = img + rng.standard_normal(img.shape)
    u, s, z, F, converged = BlakeZisserman2D(values=img).solve(0.01, 2.0, 1.0, 30.0)
    assert converged
    assert u.shape == (I, J, channels or 1)
    assert s.shape == z.shape == (I, J)
    # the discontinuity indicator s drops near the edge, stays ~1 away from it
    assert s[:, J // 2 - 1: J // 2 + 1].min() < 0.5
    assert s[:, :J // 4].mean() > 0.9


def test_block_pcg_matches_scipy_cg():
    rng = np.random.default_rng(3)
    I = J = 20
    n = I * J
    o = np.ones
    Dx = kron(eye(J), spdiags([-o(I), o(I)], [0, 1], I, I), format="csr")
    Dy = kron(spdiags([-o(J), o(J)], [0, 1], J, J), eye(I), format="csr")
    Dxx = kron(eye(J), spdiags([o(I), -2 * o(I), o(I)], [-1, 0, 1], I, I), format="csr")
    Dyy = kron(spdiags([o(J), -2 * o(J), o(J)], [-1, 0, 1], J, J), eye(I), format="csr")
    Dxy = (Dy @ Dx).tocsr()
    terms = [(Dxx, 60., 0), (Dyy, 60., 0), (Dxy, 120., 0), (Dx, .6, 1), (Dy, .6, 1)]
    M = core._DiaMatrix(terms, n, 0.02)
    w0, w1 = rng.random(n), rng.random(n)
    diag, _ = M.set_weights(w0, w1)

    # same matrix assembled with scipy
    A = sum(c * (D.T @ D.multiply((w0 if g == 0 else w1)[:, None])) for D, c, g in terms) + 0.02 * eye(n)
    B = rng.standard_normal((n, 3))
    X, info = core._block_pcg(M, 1 / diag, B, np.zeros((n, 3)), 1e-6, 10000)
    Minv = sla.LinearOperator(A.shape, matvec=lambda v: v / diag)
    for c in range(3):
        x, inf = sla.cg(A, B[:, c], rtol=1e-6, maxiter=10000, M=Minv)
        assert info[c] == inf == 0
        # matrices are assembled in a different summation order -> rounding-level differences
        assert np.linalg.norm(X[:, c] - x) <= 1e-6 * np.linalg.norm(x)
