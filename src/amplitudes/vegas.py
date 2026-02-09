from __future__ import annotations
import numpy as np

def vegas_integrate(f, ndim: int, neval: int, niter: int = 5, nbins: int = 50, rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng()
    edges = np.linspace(0.0, 1.0, nbins + 1)
    grid = np.tile(edges[None, :], (ndim, 1))

    def sample(n):
        u = rng.random((n, ndim))
        x = np.zeros_like(u)
        w = np.ones(n, dtype=np.float64)
        for d in range(ndim):
            g = grid[d]
            t = u[:, d] * nbins
            k = np.minimum(t.astype(int), nbins - 1)
            frac = t - k
            x[:, d] = g[k] + frac * (g[k + 1] - g[k])
            w *= (g[k + 1] - g[k]) * nbins
        return x, w

    estimates = []
    variances = []

    for _ in range(niter):
        x, w = sample(neval)
        fx = np.asarray([f(xi) for xi in x], dtype=np.float64)
        contrib = fx * w
        I = contrib.mean()
        V = contrib.var(ddof=1) / neval
        estimates.append(I)
        variances.append(V)

        for d in range(ndim):
            t = (x[:, d] * nbins).astype(int)
            t = np.clip(t, 0, nbins - 1)
            acc = np.zeros(nbins, dtype=np.float64)
            np.add.at(acc, t, np.abs(fx) * w)
            acc += 1e-30
            cdf = np.cumsum(acc)
            cdf /= cdf[-1]
            grid[d, 1:-1] = np.interp(np.linspace(0, 1, nbins + 1)[1:-1], cdf, edges[:-1])
            grid[d, 0] = 0.0
            grid[d, -1] = 1.0

    wts = np.array([1.0 / v if v > 0 else 0.0 for v in variances], dtype=np.float64)
    Ihat = float(np.sum(wts * np.array(estimates)) / np.sum(wts))
    var = float(1.0 / np.sum(wts))
    return Ihat, np.sqrt(var)
