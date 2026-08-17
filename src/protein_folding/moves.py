"""
The pivot move operator.

A *pivot move* picks a residue along the chain and rotates the whole sub-chain
after it by 90/180/270 degrees about one of the coordinate axes. It is the core
neighbourhood operator shared by Simulated Annealing, Hill Climbing and (as a
mutation) the Genetic Algorithm.
"""

import math
import random


def rotate_vector(vec, axis, angle_degrees):
    """Rotate a 3D vector by ``angle_degrees`` about the given axis ('x'/'y'/'z')."""
    x, y, z = vec
    rad = math.radians(angle_degrees)

    if axis == "x":
        return (x, y * math.cos(rad) - z * math.sin(rad), y * math.sin(rad) + z * math.cos(rad))
    elif axis == "y":
        return (x * math.cos(rad) + z * math.sin(rad), y, -x * math.sin(rad) + z * math.cos(rad))
    elif axis == "z":
        return (x * math.cos(rad) - y * math.sin(rad), x * math.sin(rad) + y * math.cos(rad), z)
    return vec


def distance(res1, res2):
    """Euclidean distance between two lattice positions."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(res1, res2)))


def pivot_move(protein, systematic=True, max_attempts=100):
    """Perform a pivot move on a protein chain, seeking a lower-energy conformation.

    Args:
        protein: a :class:`~protein_folding.lattice.LatticeProtein`.
        systematic: if True, try all nine (axis, angle) moves at random pivots and
            keep the best energy-improving valid move; falls back to random sampling
            if none is found. If False, accept the first valid move regardless of
            energy (used to guarantee progress / diversify).
        max_attempts: cap on the number of move attempts.

    Returns:
        New positions if a valid move is found, otherwise the original positions.
    """
    original_energy = protein._energy
    original_positions = list(protein._positions)
    best_energy = original_energy
    best_positions = None

    n = protein._length

    if systematic:
        moves = [
            ("x", 90), ("x", 180), ("x", 270),
            ("y", 90), ("y", 180), ("y", 270),
            ("z", 90), ("z", 180), ("z", 270),
        ]

        attempts = 0
        while attempts < max_attempts and best_positions is None:
            pivot_idx = random.randint(1, n - 2)

            for axis, angle in moves:
                attempts += 1
                if attempts >= max_attempts:
                    break

                # Reset to the original conformation for each candidate move.
                protein._positions = list(original_positions)
                pivot_pos = protein._positions[pivot_idx]

                # Rotate every residue from the pivot onward about the pivot.
                for i in range(pivot_idx, n):
                    rel = tuple(a - b for a, b in zip(protein._positions[i], pivot_pos))
                    rotated = rotate_vector(rel, axis, angle)
                    protein._positions[i] = tuple(round(pivot_pos[j] + rotated[j]) for j in range(3))

                if _is_valid(protein._positions, n):
                    new_energy = protein.calculate_energy()
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_positions = list(protein._positions)

                # Restore for the next attempt.
                protein._positions = list(original_positions)
                protein._energy = original_energy

        if best_positions is not None:
            return best_positions
        # Systematic search found nothing better; fall back to random sampling.
        return pivot_move(protein, systematic=False, max_attempts=max_attempts)

    # Random sampling: accept the first valid move to guarantee progress.
    attempts = 0
    while attempts < max_attempts:
        attempts += 1

        pivot_idx = random.randint(1, n - 2)
        axis = random.choice(["x", "y", "z"])
        angle = random.choice([90, 180, 270])

        temp_positions = list(original_positions)
        pivot_pos = temp_positions[pivot_idx]
        for i in range(pivot_idx, n):
            rel = tuple(a - b for a, b in zip(temp_positions[i], pivot_pos))
            rotated = rotate_vector(rel, axis, angle)
            temp_positions[i] = tuple(round(pivot_pos[j] + rotated[j]) for j in range(3))

        if _is_valid(temp_positions, n):
            return list(temp_positions)

        if best_positions is None:
            best_positions = list(temp_positions)

    return best_positions if best_positions is not None else original_positions


def _is_valid(positions, n):
    """A conformation is valid if it is self-avoiding and unit-connected."""
    if len(set(positions)) != n:
        return False
    for i in range(1, n):
        if distance(positions[i], positions[i - 1]) != 1:
            return False
    return True
