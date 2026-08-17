"""
Command-line runner for the lattice protein folding metaheuristics.

Examples:
    python main.py --pdb data/2a3d.pdb --algo all --time-limit 60
    python main.py --algo sa --time-limit 30 --seed 21 --save-plots

Run ``python main.py --help`` for all options.
"""

import argparse
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from protein_folding import (  # noqa: E402
    get_hp,
    simulated_annealing,
    genetic_algorithm,
    ant_colony_optimisation,
    hill_climbing,
    plotting,
)

SOLVERS = {
    "sa": ("Simulated Annealing", simulated_annealing),
    "ga": ("Genetic Algorithm", genetic_algorithm),
    "aco": ("Ant Colony Optimisation", ant_colony_optimisation),
    "hc": ("Hill Climbing", hill_climbing),
}


def run_solver(key, hp_array, residues, time_limit, save_dir):
    name, solver = SOLVERS[key]
    print(f"\n=== {name} ===")
    start = time.time()
    protein, energy_history, position_history = solver(hp_array, residues, time_limit=time_limit)
    elapsed = time.time() - start

    print(f"Initial energy: {energy_history[0]}")
    print(f"Final energy:   {protein.energy}")
    print(f"Runtime:        {elapsed:.2f} s")

    struct_path = os.path.join(save_dir, f"{key}_structure.png") if save_dir else None
    plotting.plot_protein_structure_3d(name, protein, "Final Optimised Structure ", save_path=struct_path)
    return name, energy_history, elapsed


def plot_convergence(results, save_dir):
    """Overlay each solver's best-energy-so-far against iteration count."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, energy_history, _ in results:
        best_so_far = np.minimum.accumulate(energy_history)
        ax.plot(range(len(best_so_far)), best_so_far, linewidth=2, alpha=0.8, label=name)
    ax.set_title("Energy convergence (best-so-far vs iteration)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy (lower is better)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, "convergence.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Fold a protein on a 3D lattice with metaheuristics.")
    parser.add_argument("--pdb", default="data/2a3d.pdb", help="path to the input PDB file")
    parser.add_argument("--algo", choices=list(SOLVERS) + ["all"], default="all",
                        help="which solver to run (default: all)")
    parser.add_argument("--time-limit", type=int, default=60,
                        help="seconds allotted to each solver (default: 60)")
    parser.add_argument("--seed", type=int, default=21, help="RNG seed (default: 21)")
    parser.add_argument("--save-plots", action="store_true",
                        help="save figures under docs/images/ instead of showing them")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.pdb):
        parser.error(f"PDB file not found: {args.pdb}\n"
                     f"Download it with: python download_pdb.py {os.path.splitext(os.path.basename(args.pdb))[0]}")

    save_dir = None
    if args.save_plots:
        save_dir = os.path.join("docs", "images")
        os.makedirs(save_dir, exist_ok=True)

    print(f"Loading protein from {args.pdb} ...")
    hp_array, residues = get_hp(args.pdb)
    print(f"HP sequence: {''.join(hp_array)}")
    print(f"Length: {len(hp_array)} residues "
          f"(H: {sum(a == 'H' for a in hp_array)}, P: {sum(a == 'P' for a in hp_array)})")

    keys = list(SOLVERS) if args.algo == "all" else [args.algo]
    results = [run_solver(k, hp_array, residues, args.time_limit, save_dir) for k in keys]

    if len(results) > 1:
        plot_convergence(results, save_dir)

    print("\n=== Summary ===")
    for name, energy_history, elapsed in results:
        print(f"{name:28s} final energy {min(energy_history):8.2f}   ({elapsed:.1f} s)")


if __name__ == "__main__":
    main()
