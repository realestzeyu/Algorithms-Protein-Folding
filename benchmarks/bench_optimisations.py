"""
Benchmark the two efficiency optimisations applied to the refactored codebase.

After studying data structures & algorithms I profiled the solvers and applied two
changes that leave results **identical** but make the code much faster:

  1. MJ energy lookup: pandas ``DataFrame.loc[a, b]``  ->  a ``dict`` hash map.
  2. Copying conformations: ``copy.deepcopy(positions)``  ->  a shallow ``list(...)``
     copy (coordinates are immutable tuples, so a deep copy was never needed).

This script measures both at four levels and asserts the energies are unchanged, so
every speedup is provably from the data structures, not from changing behaviour:

    1. raw MJ lookup            (dict vs DataFrame.loc)
    2. copying a conformation   (list() vs copy.deepcopy)
    3. one calculate_energy call (dict vs .loc, identical energy)
    4. a real solver, fixed iterations: original (deepcopy + .loc) vs optimised
       (shallow copy + dict) -- same work, same final energy, less time.

Run from the repo root:

    python benchmarks/bench_optimisations.py
"""

import copy
import os
import statistics
import sys
import time
import timeit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from protein_folding import algorithms, moves  # noqa: E402
from protein_folding.mj_matrix import mj_matrix, MJ_ENERGY  # noqa: E402
from protein_folding.lattice import LatticeProtein, get_hp  # noqa: E402
from protein_folding.moves import rotate_vector, _is_valid  # noqa: E402
from protein_folding.algorithms import hill_climbing  # noqa: E402

PDB = os.path.join(os.path.dirname(__file__), "..", "data", "2a3d.pdb")
SEED = 21


def _mean_std(values, scale):
    scaled = [v * scale for v in values]
    m = statistics.mean(scaled)
    s = statistics.stdev(scaled) if len(scaled) > 1 else 0.0
    return m, s


# --------------------------------------------------------------------------- #
# 1. Raw MJ lookup
# --------------------------------------------------------------------------- #
def bench_lookup(number=200_000, repeat=7):
    print("1) MJ lookup      DataFrame.loc[a,b]  vs  dict[(a,b)]")
    assert mj_matrix.loc["GLN", "LYS"] == MJ_ENERGY[("GLN", "LYS")]
    loc, _ = _mean_std([t / number for t in timeit.repeat(
        lambda: mj_matrix.loc["GLN", "LYS"], number=number, repeat=repeat)], 1e6)
    dct, _ = _mean_std([t / number for t in timeit.repeat(
        lambda: MJ_ENERGY[("GLN", "LYS")], number=number, repeat=repeat)], 1e6)
    print(f"   DataFrame.loc : {loc:8.3f} us")
    print(f"   dict          : {dct:8.3f} us   ->  {loc / dct:.0f}x faster\n")


# --------------------------------------------------------------------------- #
# 2. Copying a conformation
# --------------------------------------------------------------------------- #
def bench_copy(hp, res, number=100_000, repeat=5):
    print("2) copy positions   copy.deepcopy(pos)  vs  list(pos)")
    pos = LatticeProtein(hp, res)._positions
    dc, _ = _mean_std([t / number for t in timeit.repeat(
        lambda: copy.deepcopy(pos), number=number, repeat=repeat)], 1e6)
    sc, _ = _mean_std([t / number for t in timeit.repeat(
        lambda: list(pos), number=number, repeat=repeat)], 1e6)
    print(f"   copy.deepcopy : {dc:8.3f} us")
    print(f"   list()        : {sc:8.3f} us   ->  {dc / sc:.0f}x faster\n")


# --------------------------------------------------------------------------- #
# 3. One calculate_energy evaluation
# --------------------------------------------------------------------------- #
def _energy_loc(protein):
    """calculate_energy using DataFrame.loc (the pre-refactor version)."""
    energy = 0
    for i in range(protein._length):
        for j in range(i + 1, protein._length):
            if protein.is_adjacent(protein._positions[i], protein._positions[j]) and abs(i - j) > 1:
                try:
                    energy += mj_matrix.loc[protein._seq[i], protein._seq[j]]
                except KeyError:
                    continue
    return energy


def bench_calculate_energy(hp, res, number=300, repeat=5):
    print("3) calculate_energy   .loc vs dict   (whole O(n^2) evaluation)")
    protein = LatticeProtein(hp, res)
    assert _energy_loc(protein) == protein.calculate_energy(), "energies differ!"
    loc, _ = _mean_std([t / number for t in timeit.repeat(
        lambda: _energy_loc(protein), number=number, repeat=repeat)], 1e3)
    dct, _ = _mean_std([t / number for t in timeit.repeat(
        lambda: protein.calculate_energy(), number=number, repeat=repeat)], 1e3)
    print(f"   DataFrame.loc : {loc:8.3f} ms")
    print(f"   dict          : {dct:8.3f} ms   ->  {loc / dct:.2f}x faster")
    print(f"   (identical energy: {protein.calculate_energy()})\n")


# --------------------------------------------------------------------------- #
# 4. End-to-end solver: original vs optimised, same fixed iterations
# --------------------------------------------------------------------------- #
def _slow_pivot(protein, systematic=True, max_attempts=100):
    """The ORIGINAL pivot_move: identical logic but with copy.deepcopy (pre-optimisation)."""
    orig_e = protein._energy
    orig = copy.deepcopy(protein._positions)
    best_e, best, n = orig_e, None, protein._length
    if systematic:
        moves_list = [(a, ang) for a in "xyz" for ang in (90, 180, 270)]
        att = 0
        while att < max_attempts and best is None:
            piv = __import__("random").randint(1, n - 2)
            for ax, an in moves_list:
                att += 1
                if att >= max_attempts:
                    break
                protein._positions = copy.deepcopy(orig)
                pp = protein._positions[piv]
                for i in range(piv, n):
                    rel = tuple(a - b for a, b in zip(protein._positions[i], pp))
                    r = rotate_vector(rel, ax, an)
                    protein._positions[i] = tuple(round(pp[j] + r[j]) for j in range(3))
                if _is_valid(protein._positions, n):
                    ne = protein.calculate_energy()
                    if ne < best_e:
                        best_e, best = ne, copy.deepcopy(protein._positions)
                protein._positions = copy.deepcopy(orig)
                protein._energy = orig_e
        return best if best is not None else _slow_pivot(protein, False, max_attempts)
    rnd = __import__("random")
    att = 0
    while att < max_attempts:
        att += 1
        piv = rnd.randint(1, n - 2)
        ax = rnd.choice("xyz")
        an = rnd.choice([90, 180, 270])
        tp = copy.deepcopy(orig)
        pp = tp[piv]
        for i in range(piv, n):
            rel = tuple(a - b for a, b in zip(tp[i], pp))
            r = rotate_vector(rel, ax, an)
            tp[i] = tuple(round(pp[j] + r[j]) for j in range(3))
        if _is_valid(tp, n):
            return copy.deepcopy(tp)
        if best is None:
            best = copy.deepcopy(tp)
    return best if best is not None else orig


def _run_hc(hp, res, iters, seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    start = time.time()
    protein, energy_history, _ = hill_climbing(hp, res, time_limit=10 ** 9, max_iter=iters)
    return time.time() - start, len(energy_history) - 1, round(protein.energy, 4)


def bench_solver(hp, res, iters=120):
    print(f"4) Hill Climbing, {iters} fixed iterations: ORIGINAL vs OPTIMISED (same seed)")
    real_pivot = moves.pivot_move
    real_energy = LatticeProtein.calculate_energy
    try:
        # ORIGINAL: deepcopy-based pivot + DataFrame.loc energy.
        algorithms.pivot_move = _slow_pivot
        LatticeProtein.calculate_energy = lambda self: _energy_loc(self)
        t_orig, _, e_orig = _run_hc(hp, res, iters, SEED)
    finally:
        algorithms.pivot_move = real_pivot
        LatticeProtein.calculate_energy = real_energy

    # OPTIMISED: shallow copy + dict (the shipped code).
    t_opt, _, e_opt = _run_hc(hp, res, iters, SEED)

    assert e_orig == e_opt, f"energies differ! {e_orig} vs {e_opt}"
    print(f"   ORIGINAL  (deepcopy + DataFrame.loc): {t_orig:6.2f} s")
    print(f"   OPTIMISED (list() copy + dict)      : {t_opt:6.2f} s")
    print(f"   -> {t_orig / t_opt:.1f}x faster for the same {iters} iterations "
          f"(identical energy {e_opt}), i.e. ~{t_orig / t_opt:.1f}x more iterations "
          f"in the same wall-clock time.\n")


def main():
    print(f"Loading {os.path.relpath(PDB)} ...")
    hp, res = get_hp(PDB)
    print(f"{len(res)} residues\n")
    bench_lookup()
    bench_copy(hp, res)
    bench_calculate_energy(hp, res)
    bench_solver(hp, res)
    print("All correctness checks PASSED — every optimisation returns identical "
          "energies, just faster.")


if __name__ == "__main__":
    main()
