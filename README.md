# 📊 Protein Folding with Metaheuristic Algorithms

> ⚠️ **Disclaimer**  
> Due to the large file size, there is one optimised version for viewing on GitHub and one downloadable version.
> Please read the report for better understanding of what we are doing in the ipynb file.
> 
> [ Viewing Notebook (Optimised)](https://github.com/realestzeyu/Algorithms-Protein-Folding/blob/main/Protein%20Folding%20Main%20File%20with%202a3d%20Optimised.ipynb)  
> [ Downloadable Notebook (Full Version)](https://github.com/realestzeyu/Algorithms-Protein-Folding/blob/main/Protein%20Folding%20Main%20File%20with%202a3d.ipynb)
> 
> Running the algorithm gets different results everytime, thus don't be surprised when the graph does not match the on in report.
> 
> _This readme is HEAVILY simplifed, for full details, refer to the PDF report in this repository:_ [📄 Project Report (PDF)](https://github.com/realestzeyu/Algorithms-Protein-Folding/blob/main/Metaheuristics%20Report%20for%20Github.pdf)

> 🧹 **Two versions of the code**
> - The **`.ipynb` notebooks** are the **original** version — the research and the write-up live here.
> - [**`src/protein_folding/`**](src/protein_folding) is a **cleaner, refactored** version of the exact same logic: an importable Python package you can run from the command line. Same algorithms, just reorganised. See [Refactored Codebase](#️-refactored-codebase-cleaner-version) below for usage.

> 🤖 **On AI assistance:** The original project — the research, algorithms, notebooks, and report — is entirely my own work. [Claude Code](https://claude.com/claude-code) was used **only** to refactor that existing code into the `src/` package and add supporting tooling (CLI, benchmarks). The methods and results are unchanged.


## Introduction  
Protein folding is the process where a linear chain of amino acids (the **primary structure**) spontaneously collapses into a stable 3D conformation (the **tertiary structure**).  
Even with the complete amino acid sequence, predicting this final structure remains an **NP-Hard problem** due to the **astronomical number of possible configurations** and complex interactions among residues.

We explored how **metaheuristic algorithms** can be applied to this problem.  
Existing solutions (e.g., the **HP lattice model**) simplify the problem using binary hydrophobic/polar classifications, but often ignore deeper energetic interactions.

To enhance this, we incorporated the **Miyazawa–Jernigan (MJ) energy matrix**, providing a more chemically realistic interaction model.

---

## Refactored Codebase (cleaner version)

The original notebooks are kept exactly as they are. The same logic has also been
refactored into a clean, importable Python package under
[`src/protein_folding/`](src/protein_folding) so it can be run from the command line
instead of only inside Jupyter.

```
├── main.py                 # CLI runner
├── download_pdb.py         # fetch any structure from the RCSB PDB
├── requirements.txt
├── data/2a3d.pdb           # example input
└── src/protein_folding/
    ├── mj_matrix.py        # Miyazawa–Jernigan contact-energy matrix
    ├── lattice.py          # LatticeProtein model + PDB/HP sequence loading
    ├── moves.py            # pivot-move operator
    ├── algorithms.py       # Simulated Annealing, GA, ACO, Hill Climbing
    └── plotting.py         # 3D structure & convergence plots
```

**Usage**

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. get an input protein (2a3d already ships in data/; any PDB ID works)
python download_pdb.py 2a3d

# 3. run the solvers
python main.py --pdb data/2a3d.pdb --algo all --time-limit 60
```

Options: `--algo {sa,ga,aco,hc,all}`, `--time-limit <seconds>`, `--seed <int>`,
and `--save-plots` (save figures instead of opening windows).

Or use it as a library:

```python
from protein_folding import get_hp, simulated_annealing

hp_seq, residues = get_hp("data/2a3d.pdb")
protein, energy_history, position_history = simulated_annealing(hp_seq, residues, time_limit=60)
print(protein.energy)
```

> The refactor preserves the original algorithms — results are still stochastic and
> will differ between runs. See the PDF report for the recorded findings.

### ⚡ Efficiency: profiling & optimising the hot path

After studying **data structures & algorithms**, I profiled the solvers and found the two
biggest bottlenecks were both in the per-move inner loop. Fixing them makes the refactored
code **~2.3× faster end-to-end** (measured on Hill Climbing, `2a3d`) while producing
**byte-identical energies** — the speed comes purely from better data structures, not from
changing the algorithms.

**1. MJ energy lookup → hash map.** The original code reads energies with
`mj_matrix.loc[a, b]` on a pandas DataFrame, inside `calculate_energy`, which runs
**O(n²)** on *every* move. Switching the lookup to a `dict` keyed by residue pair makes
each lookup **~77× faster** and `calculate_energy` **~1.9× faster**. The package keeps the
DataFrame for display but uses the hash map (`MJ_ENERGY`) in the hot loops.

**2. Removing unnecessary `deepcopy`.** Profiling showed `copy.deepcopy` was **~42%** of
the solver runtime — the code was deep-copying conformations (lists of coordinate
*tuples*) constantly. Since tuples are immutable, a shallow `list(...)` copy is correct
and **~465× faster** per copy. Replacing the deep copies (hottest one is in `pivot_move`)
removed that entire bottleneck.

Reproduce it yourself:

```bash
python benchmarks/bench_optimisations.py
```

It compares the original vs optimised approach at four levels — raw lookup, copying a
conformation, one full `calculate_energy` call, and a fixed-iteration Hill Climbing run —
and **asserts the energies are identical**, so every speedup is provably behaviour-neutral.

---

## Dataset Overview  
- **Source**: Protein Data Bank (PDB)  
- **Example**: [`2a3d.pdb`](https://www.rcsb.org/structure/2a3d)  
- Note: Users may download any `.pdb` file of interest for folding experiments, simply change the pdb file name at the end of notebook

---

## Methodology  

We implemented and benchmarked several metaheuristic approaches:

1. **Custom Pivot Move Operator**  
  Inspired by techniques in game development for simulating physical transformations.

2. **Object-Oriented Design**  
  Built a flexible `LatticeProtein` class to manage structure, energy, and move operations.

3. **Heuristic & Metaheuristic Algorithms**
  - Greedy heuristic baseline  
  - **Simulated Annealing (SA)**  
  - **Hill Climbing (HC)**  
  - **Genetic Algorithm (GA)**  
  - **Ant Colony Optimisation (ACO)**

4. **Benchmarking & Evaluation**  
  We tested each algorithm's performance based on energy minimisation and folding accuracy.

---

## Key Findings  
> _Summary of results and insights are available in the PDF report._ [📄 Project Report (PDF)](https://github.com/realestzeyu/Algorithms-Protein-Folding/blob/main/Metaheuristics%20Report%20for%20Github.pdf)

