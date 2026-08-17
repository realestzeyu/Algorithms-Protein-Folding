"""
The lattice protein model.

A protein is represented as a self-avoiding walk on the 3D cubic lattice: each
residue sits on an integer coordinate, consecutive residues are unit-distance
neighbours, and no two residues share a cell. The energy of a conformation is
the sum of MJ contact energies over all non-sequential residue pairs that end up
adjacent on the lattice.
"""

import numpy as np
from Bio.PDB import PDBParser

from .mj_matrix import MJ_ENERGY

# Residues classified as hydrophobic (H); everything else is polar (P).
# This is used for the HP colouring/heuristics; the *energy* itself always uses
# the full residue identity via the MJ matrix.
HYDROPHOBIC = {"ALA", "VAL", "ILE", "LEU", "PHE", "MET", "TRP"}

# The six unit moves on the cubic lattice.
MOVES = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def aa_to_hp(sequence):
    """Map a sequence of 3-letter residue codes to an array of 'H'/'P' labels."""
    return np.array(["H" if aa.upper() in HYDROPHOBIC else "P" for aa in sequence])


def get_hp(pdb_file):
    """Parse a PDB file and return ``(hp_array, residues)``.

    ``residues`` is the list of 3-letter residue codes (the full sequence used
    for energy), and ``hp_array`` is the matching H/P classification.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    residues = []
    for res in structure.get_residues():
        # A blank hetero-flag (res.id[0] == " ") marks a standard residue.
        if res.id[0] == " ":
            residues.append(res.get_resname())
    return aa_to_hp(residues), residues


class LatticeProtein:
    """A protein conformation on the 3D cubic lattice.

    Construction seeds a greedy self-avoiding walk (:meth:`initial_solution`)
    and evaluates its energy. ``positions`` and ``energy`` expose the current
    state; the underlying ``_positions`` / ``_energy`` attributes are still
    written directly by the optimisation algorithms.
    """

    def __init__(self, hp_sequence, original_seq):
        self._hp_seq = list(hp_sequence)
        self._seq = list(original_seq)  # full residue identity, drives energy
        self._length = len(original_seq)
        self._positions = self.initial_solution()
        self._energy = self.calculate_energy()

    @property
    def positions(self):
        """List of ``(x, y, z)`` integer lattice coordinates, one per residue."""
        return self._positions

    @property
    def energy(self):
        """Total MJ contact energy of the current conformation (lower is better)."""
        return self._energy

    @property
    def hp_sequence(self):
        return self._hp_seq

    @property
    def sequence(self):
        return self._seq

    def clone(self):
        """Return an independent copy — a fast replacement for ``copy.deepcopy``.

        Positions are shallow-copied (they are lists of immutable int tuples, so a
        shallow copy is safe); the never-mutated residue lists are shared. Uses
        ``__new__`` to skip the greedy ``__init__`` rebuild.
        """
        new = LatticeProtein.__new__(LatticeProtein)
        new._hp_seq = self._hp_seq
        new._seq = self._seq
        new._length = self._length
        new._positions = list(self._positions)
        new._energy = self._energy
        return new

    def initial_solution(self):
        """Greedy heuristic: place residues one at a time in the lowest-energy
        adjacent cell, producing a reasonable starting conformation."""
        if self._length == 0:
            return []

        positions = [(0, 0, 0)]  # start at the origin

        # Residue 0 is fixed; place each subsequent residue greedily.
        for i in range(1, self._length):
            best_pos = None
            best_energy_gain = float("inf")

            # Candidate cells: any empty cell adjacent to an already-placed residue.
            candidate_positions = set()
            for existing_pos in positions:
                for move in MOVES:
                    new_pos = tuple(existing_pos[j] + move[j] for j in range(3))
                    if new_pos not in positions:
                        candidate_positions.add(new_pos)

            for candidate_pos in candidate_positions:
                # Must stay chain-connected to the previous residue.
                if not self.is_adjacent(candidate_pos, positions[i - 1]):
                    continue

                # Energy gained from new non-sequential contacts at this cell.
                energy_gain = 0
                for j in range(i):
                    if self.is_adjacent(candidate_pos, positions[j]) and abs(i - j) > 1:
                        # .get(..., 0.0) skips residue pairs absent from the matrix,
                        # matching the old "continue on KeyError" behaviour.
                        energy_gain += MJ_ENERGY.get((self._seq[i], self._seq[j]), 0.0)

                if energy_gain < best_energy_gain:
                    best_energy_gain = energy_gain
                    best_pos = candidate_pos

            # Fallback: any empty cell adjacent to the previous residue.
            if best_pos is None:
                for move in MOVES:
                    new_pos = tuple(positions[i - 1][j] + move[j] for j in range(3))
                    if new_pos not in positions:
                        best_pos = new_pos
                        break

            positions.append(best_pos)

        return positions

    def calculate_energy(self):
        """Sum MJ contact energies over all non-sequential adjacent residue pairs."""
        energy = 0
        for i in range(self._length):
            for j in range(i + 1, self._length):
                # abs(i - j) > 1 skips covalently bonded (sequential) neighbours.
                if self.is_adjacent(self._positions[i], self._positions[j]) and abs(i - j) > 1:
                    energy += MJ_ENERGY.get((self._seq[i], self._seq[j]), 0.0)
        return energy

    @staticmethod
    def is_adjacent(pos1, pos2):
        """True if the two cells are unit (Manhattan) distance apart."""
        return (
            abs(pos1[0] - pos2[0])
            + abs(pos1[1] - pos2[1])
            + abs(pos1[2] - pos2[2])
        ) == 1
