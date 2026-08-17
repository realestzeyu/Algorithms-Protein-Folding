"""
Protein folding on a 3D cubic lattice with metaheuristic algorithms.

This package is a refactored, importable version of the code that originally
lived (copy-pasted) inside the project notebooks. The notebooks are kept as-is
for the write-up; this package is the clean, reusable implementation.

Typical usage:

    from protein_folding import get_hp, simulated_annealing

    hp_seq, residues = get_hp("data/2a3d.pdb")
    protein, energy_history, position_history = simulated_annealing(
        hp_seq, residues, time_limit=60
    )
    print(protein.energy)
"""

from .mj_matrix import mj_matrix, MJ_ENERGY
from .lattice import LatticeProtein, aa_to_hp, get_hp
from .moves import pivot_move, rotate_vector, distance
from .algorithms import (
    simulated_annealing,
    genetic_algorithm,
    ant_colony_optimisation,
    hill_climbing,
)
from . import plotting

__all__ = [
    "mj_matrix",
    "MJ_ENERGY",
    "LatticeProtein",
    "aa_to_hp",
    "get_hp",
    "pivot_move",
    "rotate_vector",
    "distance",
    "simulated_annealing",
    "genetic_algorithm",
    "ant_colony_optimisation",
    "hill_climbing",
    "plotting",
]
