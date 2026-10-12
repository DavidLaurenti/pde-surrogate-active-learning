# Neural surrogate of a parametric diffusion PDE with active learning

Steady diffusion −∇·(κ(x; μ) ∇u) = f with a conductivity κ depending on a few parameters μ.
A hand-written FD/FEM solver generates high-fidelity data; a PyTorch surrogate learns μ → QoI / u;
sampling strategies (random, LHS, ensemble-based active learning) are compared; Sobol indices quantify sensitivity.

## Results
_TODO: solver convergence plot, surrogate error vs. number of simulations, Sobol indices._

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[dev]"
pytest
```

## Layout
- `src/pde_surrogate/solver/` — FD/FEM solver + convergence tests
- `src/pde_surrogate/surrogate/` — PyTorch models and training
- `src/pde_surrogate/sampling/` — random, LHS, active learning
- `src/pde_surrogate/uq/` — Sobol indices
- `notebooks/demo.ipynb`, `figures/`
