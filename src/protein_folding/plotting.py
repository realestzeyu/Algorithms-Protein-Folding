"""
3D visualisation helpers for folded conformations.

Hydrophobic (H) residues are drawn as red circles, polar (P) residues as blue
squares, with the backbone as a connecting line. Every helper accepts an
optional ``save_path`` so figures can be written to disk (used by ``main.py``
to generate the images embedded in the README) instead of shown interactively.
"""

import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import PDBParser
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from .lattice import LatticeProtein, get_hp


def _finish(fig, save_path):
    """Either save the figure to ``save_path`` or show it interactively."""
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _draw_conformation(ax, positions, hp_sequence, original_seq, label=True):
    if not positions:
        return
    xs, ys, zs = zip(*positions)
    ax.plot(xs, ys, zs, "-", color="gray", linewidth=2, alpha=0.6)
    for j, (x, y, z) in enumerate(positions):
        color = "red" if hp_sequence[j] == "H" else "blue"
        marker = "o" if hp_sequence[j] == "H" else "s"
        ax.scatter(x, y, z, c=color, s=10, marker=marker, alpha=0.8)
        if label:
            ax.text(x, y, z, f"{original_seq[j]}-{hp_sequence[j]}", fontsize=8, alpha=0.8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)
    ax.set_box_aspect([1, 1, 1])


def plot_last_5_iterations(which_algo, position_history, hp_sequence, original_seq, save_path=None):
    """Plot the final five recorded conformations side by side."""
    last_5 = position_history if len(position_history) < 5 else position_history[-5:]
    fig = plt.figure(figsize=(20, 4))
    for i, positions in enumerate(last_5):
        ax = fig.add_subplot(1, 5, i + 1, projection="3d")
        _draw_conformation(ax, positions, hp_sequence, original_seq)
        start_idx = len(position_history) - len(last_5) + i
        ax.set_title(f"{which_algo}--Iter {start_idx + 1}\n(Red circles=H, Blue squares=P)")
    _finish(fig, save_path)


def plot_first_and_last_iterations(which_algo, position_history, hp_sequence, original_seq, save_path=None):
    """Compare the initial (greedy) conformation against the final best one."""
    iterations = [LatticeProtein(hp_sequence, original_seq)._positions, position_history[-1]]
    labels = ["Iter 1", f"Iter {len(position_history)}"]
    fig = plt.figure(figsize=(12, 4))
    for i, positions in enumerate(iterations):
        ax = fig.add_subplot(1, len(iterations), i + 1, projection="3d")
        _draw_conformation(ax, positions, hp_sequence, original_seq)
        ax.set_title(f"{which_algo}--{labels[i]}\n(Red circles=H, Blue squares=P)")
    _finish(fig, save_path)


def plot_protein_structure_3d(which_algo, protein, title="Protein Structure", save_path=None):
    """Render a single folded conformation with a legend and energy annotation."""
    positions = protein._positions
    hp_sequence = protein._hp_seq
    residue_chain = protein._seq

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if positions:
        xs, ys, zs = zip(*positions)
        ax.plot(xs, ys, zs, "-", color="gray", linewidth=3, alpha=0.7, label="Backbone")
        h_count = p_count = 0
        for i, (x, y, z) in enumerate(positions):
            if hp_sequence[i] == "H":
                ax.scatter(x, y, z, c="red", s=150, alpha=0.9, edgecolors="darkred", linewidth=2)
                h_count += 1
            else:
                ax.scatter(x, y, z, c="blue", s=150, alpha=0.9, edgecolors="darkblue", linewidth=2)
                p_count += 1
            ax.text(x, y, z + 0.2, f"{residue_chain[i]}-{hp_sequence[i]}",
                    fontsize=10, ha="center", va="bottom", weight="bold")
        ax.scatter([], [], c="red", s=150, alpha=0.9, edgecolors="darkred", linewidth=2,
                   label=f"Hydrophobic (H) - {h_count}")
        ax.scatter([], [], c="blue", s=150, alpha=0.9, edgecolors="darkblue", linewidth=2,
                   label=f"Polar (P) - {p_count}")

    ax.set_title(f"{title}{which_algo}\nEnergy: {protein._energy}", fontsize=14, weight="bold")
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_box_aspect([1, 1, 1])
    _finish(fig, save_path)


def plot_original_pdb(pdb_file, save_path=None):
    """Plot the experimental C-alpha trace from a PDB file for comparison."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("original", pdb_file)

    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " " and "CA" in residue:
                    ca_coords.append(residue["CA"].get_coord())
    ca_coords = np.array(ca_coords)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2],
            "k-", alpha=0.5, linewidth=1, label="Backbone")

    hp_sequence, residue_chain = get_hp(pdb_file)
    for i, (x, y, z) in enumerate(ca_coords):
        color = "darkred" if hp_sequence[i] == "H" else "darkblue"
        ax.scatter(x, y, z, c=color, s=50, depthshade=False)
        ax.text(x, y, z + 0.2, f"{residue_chain[i]}-{hp_sequence[i]}",
                fontsize=10, ha="center", va="bottom", weight="bold")

    ax.set_title(f"Original PDB Structure ({structure.id})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    _finish(fig, save_path)
