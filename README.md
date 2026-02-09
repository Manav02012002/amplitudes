# amplitudes

Standalone Python library for modern on-shell scattering amplitudes:
- spinor-helicity numerics
- BCFW recursion for tree-level color-ordered gluon amplitudes
- Parke–Taylor MHV
- exact tree-level SU(N) color sums using Fierz identity (no adjoint brute force)
- RAMBO + VEGAS integration and a cross section pipeline

## Install
pip install -e ".[dev]"

## Quick sanity
pytest -q

## Demo
amplitudes demo --n 4 --seed 1

## Color-ordered amplitude (BCFW)
amplitudes partial --hels "-- -- ++" --n 4 --seed 2

## Full color-dressed |M|^2 (exact SU(N))
amplitudes me2 --hels "-- -- ++" --n 4 --nc 3 --seed 2

## Cross section (2->n in COM, massless)
amplitudes xsec --hels "-- -- ++" --n 4 --nc 3 --ecm 1000 --neval 20000 --niter 5 --seed 3


## qqbar + ng
amplitudes me2q --hels "- -- ++ -" --ecm 1000 --nc 3
amplitudes xsecqq --ng 4 --ecm 1000 --nc 3 --neval 20000 --niter 5

## gg -> ng with helicity sums
amplitudes xsecgg --ng 4 --ecm 1000 --nc 3 --neval 20000 --niter 5


## Crossing-safe 2→n
amplitudes xsec2n --init "g g" --final "g g g g" --ecm 1000 --nc 3
amplitudes xsec2n --init "q qb" --final "g g g g" --ecm 1000 --nc 3
amplitudes xsec2n --init "q qb" --final "v g g" --ecm 1000 --nc 3 --ew --flavor u


## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```
