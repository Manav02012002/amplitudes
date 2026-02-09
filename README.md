# amplitudes - spinor-helicity amplitudes and cross sections in Python

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/License-MIT-gold)
![tests](https://img.shields.io/github/actions/workflow/status/<ORG>/<REPO>/ci.yml?label=tests)
![code-style](https://img.shields.io/badge/code%20style-ruff-black)
![typecheck](https://img.shields.io/badge/typecheck-pyright-6b5cff)

`amplitudes` is a lightweight, production-ready toolkit for **tree-level scattering amplitudes** using
**spinor-helicity**, **Parke–Taylor** and **BCFW recursion**, with **color handling** and **Monte Carlo**
(phase space + VEGAS) integration for **cross sections**. It includes a CLI, a test suite, and a clean Python API.

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
