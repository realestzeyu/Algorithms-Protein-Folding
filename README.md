# 📊 Protein Folding with Metaheuristic Algorithms

> ⚠️ **Disclaimer**  
> Due to the large file size, there is one optimised version for viewing on GitHub and one downloadable version.
> Please read the report for better understanding of what we are doing in the ipynb file.
> 
> [ Viewing Notebook (Optimised)](https://github.com/realestzeyu/Algorithms-Protein-Folding/blob/main/Protein%20Folding%20Main%20File%20with%202a3d%20Optimised.ipynb)  
> [ Downloadable Notebook (Full Version)](https://github.com/realestzeyu/Algorithms-Protein-Folding/blob/main/Protein%20Folding%20Main%20File%20with%202a3d.ipynb)
> Running the algorithm gets different results everytime, thus don't be surprised when the graph does not match the on in report
> _This readme is HEAVILY simplifed, for full details, refer to the PDF report in this repository:_ [📄 Project Report (PDF)](https://github.com/realestzeyu/Algorithms-Protein-Folding/blob/main/Metaheuristics%20Report%20for%20Github.pdf)


## 📌 Introduction  
Protein folding is the process where a linear chain of amino acids (the **primary structure**) spontaneously collapses into a stable 3D conformation (the **tertiary structure**).  
Even with the complete amino acid sequence, predicting this final structure remains an **NP-Hard problem** due to the **astronomical number of possible configurations** and complex interactions among residues.

We explored how **metaheuristic algorithms** can be applied to this problem.  
Existing solutions (e.g., the **HP lattice model**) simplify the problem using binary hydrophobic/polar classifications, but often ignore deeper energetic interactions.

To enhance this, we incorporated the **Miyazawa–Jernigan (MJ) energy matrix**, providing a more chemically realistic interaction model.

---

## 📂 Dataset Overview  
- **Source**: Protein Data Bank (PDB)  
- **Example**: [`2a3d.pdb`](https://www.rcsb.org/structure/2a3d)  
- Note: Users may download any `.pdb` file of interest for folding experiments, simply change the pdb file name at the end of notebook

---

## ⚙️ Methodology  

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

## 📈 Key Findings  
> _Summary of results and insights are available in the PDF report._

