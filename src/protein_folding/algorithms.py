"""
Metaheuristic solvers for the lattice protein folding problem.

Every solver has the same interface: it takes the HP sequence and the full
residue sequence, runs until ``time_limit`` seconds elapse, and returns a
``(best_protein, energy_history, position_history)`` tuple where

* ``best_protein``      - the lowest-energy :class:`LatticeProtein` found,
* ``energy_history``    - energy per iteration (for convergence plots),
* ``position_history``  - the best conformation recorded per iteration.

Unlike the original notebook, the time budget is a parameter and timing is kept
in a local variable, so a solver can be called repeatedly in one session.
"""

import math
import random
import time

import numpy as np

from .lattice import LatticeProtein
from .moves import pivot_move

# Directions for the ACO self-avoiding walk.
_DIRECTIONS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


# --------------------------------------------------------------------------- #
# Simulated Annealing
# --------------------------------------------------------------------------- #
def simulated_annealing(hp_sequence, original_seq, time_limit=60, temp=1000,
                        max_iter=10**15):
    """Simulated Annealing with a two-phase cooling schedule."""
    program_starts = time.time()
    iteration = 0
    iteration_t = 0
    tp = temp

    protein = LatticeProtein(hp_sequence, original_seq)
    current_protein = protein.clone()
    best_protein = protein.clone()
    best_energy = protein._energy

    energy_history = [protein._energy]
    position_history = [list(protein._positions)]

    while time.time() - program_starts < time_limit and iteration < max_iter:
        iteration += 1

        # Two pivot moves per step empirically produce better proposals.
        trial_protein = current_protein.clone()
        trial_protein._positions = pivot_move(protein=trial_protein)
        trial_protein._positions = pivot_move(protein=trial_protein)
        trial_protein._energy = trial_protein.calculate_energy()

        # Two-phase cooling: a logarithmic schedule early, geometric later.
        if iteration <= 1000:
            if iteration_t > 25:
                tp = temp / math.log(iteration * 5000)
                iteration_t = 0
            iteration_t += 1
        else:
            tp = temp * (0.9975 ** iteration)

        current_energy = current_protein._energy
        delta_e = trial_protein._energy - best_energy
        metropolis = np.exp(-delta_e / tp)

        # Accept improving moves, or worse moves with the Metropolis probability.
        if trial_protein._energy < current_energy or (random.random() < metropolis and delta_e < 30):
            current_protein = trial_protein
            if trial_protein._energy < best_energy:
                best_energy = trial_protein._energy
                best_protein = trial_protein.clone()

        energy_history.append(current_protein._energy)
        position_history.append(list(best_protein._positions))

    return best_protein, energy_history, position_history


# --------------------------------------------------------------------------- #
# Genetic Algorithm
# --------------------------------------------------------------------------- #
def _scale_fitness(energy_list):
    """Turn energies (lower is better) into positive fitness (higher is better)."""
    shifted = energy_list - np.min(energy_list) + 1
    return 1.0 / shifted


def _is_valid_conformation(positions):
    """Self-avoiding and unit-connected check (used by crossover/mutation)."""
    if len(set(positions)) != len(positions):
        return False
    for i in range(len(positions) - 1):
        p1, p2 = positions[i], positions[i + 1]
        if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2]) != 1:
            return False
    return True


def _produce_child(parent_1, parent_2, hp_sequence, original_seq):
    """Single-point crossover; retries points until a valid child is found."""
    crossover_point = random.randint(1, len(parent_1) - 1)
    for _ in range(50):
        child_positions = parent_1[:crossover_point] + parent_2[crossover_point:]
        if _is_valid_conformation(child_positions):
            return child_positions
        crossover_point = random.randint(1, len(parent_1) - 1)

    # No valid child found: return the fitter parent.
    p1 = LatticeProtein(hp_sequence, original_seq)
    p1._positions = parent_1
    p2 = LatticeProtein(hp_sequence, original_seq)
    p2._positions = parent_2
    return parent_1 if p1.calculate_energy() < p2.calculate_energy() else parent_2


def _mutation(solution, hp_sequence, original_seq):
    """Apply 1-3 random pivot moves, keeping the result only if valid."""
    for _ in range(10):
        temp_protein = LatticeProtein(hp_sequence, original_seq)
        temp_protein._positions = list(solution)
        for _ in range(random.randint(1, 3)):
            temp_protein._positions = pivot_move(protein=temp_protein)
        if _is_valid_conformation(temp_protein._positions):
            return list(temp_protein._positions)
    return list(solution)


def genetic_algorithm(hp_sequence, original_seq, time_limit=60, pop_size=10,
                      max_iter=10**17):
    """Genetic Algorithm with pivot-move mutation, single-point crossover and elitism."""
    program_starts = time.time()
    iteration = 0

    obj_value_opt = float("inf")
    obj_value_list = []
    opt_solution = []
    opt_solution_list = []

    next_gen = []
    next_gen_dict = {}

    # Seed the population with randomised conformations plus one greedy solution.
    for i in range(pop_size - 1):
        protein = LatticeProtein(hp_sequence, original_seq)
        for _ in range(int(random.uniform(10, 30))):
            protein._positions = pivot_move(protein=protein, systematic=False)
        next_gen.append(list(protein._positions))
        next_gen_dict[i] = {"positions": list(protein._positions),
                            "energy": protein.calculate_energy()}

    greedy_protein = LatticeProtein(hp_sequence, original_seq)
    next_gen.append(list(greedy_protein._positions))
    next_gen_dict[pop_size - 1] = {"positions": list(greedy_protein._positions),
                                   "energy": greedy_protein.calculate_energy()}

    while time.time() - program_starts < time_limit and iteration < max_iter:
        prev_gen = [list(p) for p in next_gen]
        prev_gen_dict = {k: {"positions": list(v["positions"]), "energy": v["energy"]}
                         for k, v in next_gen_dict.items()}

        # Evaluate fitness of the current generation.
        fitness = []
        for i in range(pop_size):
            protein_temp = LatticeProtein(hp_sequence, original_seq)
            protein_temp._positions = next_gen_dict[i]["positions"]
            energy = protein_temp.calculate_energy()
            fitness.append(energy)
            next_gen_dict[i]["energy"] = energy

        fitness = np.array(fitness)
        scaled_fitness = _scale_fitness(fitness)

        # Build a cumulative distribution for roulette-wheel selection.
        distribution = scaled_fitness / np.sum(scaled_fitness)
        ind = range(0, pop_size)
        stacked = np.column_stack((distribution, ind))
        stacked = stacked[np.argsort(stacked[:, 0]), ]
        distribution = stacked[:, 0]
        cumulative_sum = np.cumsum(distribution)

        # Elitism: keep the top 40% of the current generation.
        elitism = 0.4
        best_index = pop_size - int(pop_size * elitism)
        best_ind = stacked[range(best_index, pop_size), 1].astype(int)

        # Roulette-wheel selection of two parent sets.
        r1 = np.random.uniform(0, 1, pop_size)
        r2 = np.random.uniform(0, 1, pop_size)

        def _select(random_values):
            indices = []
            for p in random_values:
                idx = 0
                while idx < len(cumulative_sum) and p > cumulative_sum[idx]:
                    idx += 1
                indices.append(min(idx, len(cumulative_sum) - 1))
            return indices

        ind1 = np.asarray(stacked[_select(r1), 1], dtype=int).tolist()
        selected = [next_gen[i] for i in ind1]
        f1 = [scaled_fitness[i] for i in ind1]

        ind2 = np.asarray(stacked[_select(r2), 1], dtype=int).tolist()
        mates = [next_gen[i] for i in ind2]
        f2 = [scaled_fitness[i] for i in ind2]

        # Crossover (95%) and mutation (75%) to build the offspring.
        p_cross, p_mut = 0.95, 0.75
        offspring_ls = []
        for i in range(len(selected)):
            if np.random.uniform(0, 1) < p_cross:
                offspring = _produce_child(selected[i], mates[i], hp_sequence, original_seq)
            else:
                offspring = mates[i] if f1[i] < f2[i] else selected[i]

            if np.random.uniform(0, 1) < p_mut:
                offspring = _mutation(offspring, hp_sequence, original_seq)
            offspring_ls.append(offspring)

        offspring_population_dict = {}
        for i, solution_i in enumerate(offspring_ls):
            protein_temp = LatticeProtein(hp_sequence, original_seq)
            protein_temp._positions = solution_i
            offspring_population_dict[i] = {"positions": solution_i,
                                            "energy": protein_temp.calculate_energy()}

        offspring_fitness = np.array([offspring_population_dict[i]["energy"]
                                      for i in range(len(offspring_population_dict))])
        offspring_scaled_fitness = _scale_fitness(offspring_fitness)

        stacked_offspring = np.column_stack((offspring_scaled_fitness, range(0, pop_size)))
        stacked_offspring = stacked_offspring[np.argsort(-stacked_offspring[:, 0]), ]

        # Replace the offspring's worst slots with the retained elites.
        next_gen = [list(p) for p in offspring_ls]
        next_gen_dict = {k: {"positions": list(v["positions"]), "energy": v["energy"]}
                         for k, v in offspring_population_dict.items()}
        if len(best_ind) > 0:
            ind_replace = np.array(stacked_offspring[best_index:pop_size, 1], dtype=int)
            for j, replace_idx in enumerate(ind_replace):
                if j < len(best_ind):
                    next_gen[replace_idx] = prev_gen[best_ind[j]]
                    next_gen_dict[replace_idx] = prev_gen_dict[best_ind[j]]

        obj_value = np.min(fitness)
        obj_value_list.append(obj_value)

        if obj_value < obj_value_opt:
            obj_value_opt = obj_value
            opt_solution_ind = int(np.where(fitness == fitness.min())[0][0])
            opt_solution = next_gen[opt_solution_ind]

        opt_solution_list.append(opt_solution)
        obj_value_list.append(obj_value)
        iteration += 1

    best_protein = LatticeProtein(hp_sequence, original_seq)
    best_protein._positions = opt_solution
    best_protein._energy = obj_value_opt
    return best_protein, obj_value_list, opt_solution_list


# --------------------------------------------------------------------------- #
# Hill Climbing
# --------------------------------------------------------------------------- #
def hill_climbing(hp_sequence, original_seq, time_limit=60, max_iter=10**17):
    """Steepest-descent hill climbing on the pivot-move neighbourhood."""
    program_starts = time.time()
    iteration = 0

    protein = LatticeProtein(hp_sequence, original_seq)
    best_protein = protein.clone()
    best_energy = protein._energy

    energy_history = [protein._energy]
    position_history = [list(protein._positions)]

    while time.time() - program_starts < time_limit and iteration < max_iter:
        iteration += 1
        protein = best_protein.clone()

        # Two pivot moves per step (empirically stronger than one).
        protein._positions = pivot_move(protein=protein)
        protein._positions = pivot_move(protein=protein)
        protein._energy = protein.calculate_energy()

        if protein._energy < best_energy:
            best_energy = protein._energy
            best_protein = protein.clone()

        energy_history.append(protein._energy)
        position_history.append(list(best_protein._positions))

    return best_protein, energy_history, position_history


# --------------------------------------------------------------------------- #
# Ant Colony Optimisation
# --------------------------------------------------------------------------- #
def ant_colony_optimisation(hp_sequence, original_seq, time_limit=60, max_iter=999999,
                            num_ants=15, alpha=0.4, beta=1.5, evaporation_rate=0.1,
                            initial_pheromone=1.0):
    """Ant Colony Optimisation that constructs folds directly (no pivot moves).

    Returns ``(best_protein, energy_history, position_history)`` - note this order
    matches the other solvers (the original notebook returned the last two swapped).
    """
    program_starts = time.time()
    sequence_length = len(hp_sequence)
    protein = LatticeProtein(hp_sequence, original_seq)

    pheromones = {
        step: {direction: initial_pheromone for direction in _DIRECTIONS}
        for step in range(1, len(hp_sequence))
    }

    best_energy = protein._energy
    energy_history = [best_energy]
    best_positions = list(protein._positions)
    position_history = [list(protein._positions)]
    best_protein = protein.clone()

    def calculate_enhanced_heuristic(current_pos, direction, protein_state, step):
        """Score a candidate direction: reward H-H contacts, compactness, neighbours."""
        new_pos = tuple(current_pos[i] + direction[i] for i in range(3))
        if new_pos in protein_state._positions:
            return 0.0

        heuristic_value = 0.1
        # Reward proximity to other hydrophobic residues (core formation).
        if step < len(hp_sequence) and hp_sequence[step] == "H":
            for i, pos in enumerate(protein_state._positions[:step]):
                if i < len(hp_sequence) and hp_sequence[i] == "H":
                    d = sum(abs(new_pos[j] - pos[j]) for j in range(3))
                    if d == 1:
                        heuristic_value += 5.0
                    elif d == 2:
                        heuristic_value += 1.0

        # Reward staying near the current centre of mass (compactness).
        if len(protein_state._positions) > 2:
            center = [sum(pos[i] for pos in protein_state._positions[:step]) / step for i in range(3)]
            dist_to_center = sum((new_pos[i] - center[i]) ** 2 for i in range(3)) ** 0.5
            heuristic_value += max(0, 3.0 - dist_to_center * 0.5)

        # Reward cells with more occupied neighbours.
        neighbor_count = sum(
            1 for pos in protein_state._positions[:step]
            if sum(abs(new_pos[i] - pos[i]) for i in range(3)) == 1
        )
        if neighbor_count > 1:
            heuristic_value += 2.0
        elif neighbor_count == 0:
            heuristic_value *= 0.1

        # Discourage continuing straight or doubling back.
        if step > 1:
            prev_direction = tuple(current_pos[i] - protein_state._positions[step - 2][i] for i in range(3))
            if direction == prev_direction:
                heuristic_value *= 0.3
            elif tuple(-d for d in direction) == prev_direction:
                heuristic_value *= 0.1

        return max(heuristic_value, 0.01)

    def probabilistic_selection(protein_state, step):
        """Pick the next direction using the pheromone/heuristic rule."""
        if step >= len(hp_sequence):
            return None

        current_pos = protein_state._positions[step - 1]
        valid_moves = []
        move_probabilities = []
        for direction in _DIRECTIONS:
            new_pos = tuple(current_pos[i] + direction[i] for i in range(3))
            if new_pos not in protein_state._positions:
                pheromone_level = pheromones.get(step, {}).get(direction, initial_pheromone)
                heuristic_value = calculate_enhanced_heuristic(current_pos, direction, protein_state, step)
                valid_moves.append(direction)
                move_probabilities.append((pheromone_level ** alpha) * (heuristic_value ** beta))

        if not valid_moves:
            return None

        total_prob = sum(move_probabilities)
        if total_prob == 0:
            return random.choice(valid_moves)

        probabilities = [p / total_prob for p in move_probabilities]
        return valid_moves[np.random.choice(len(valid_moves), p=probabilities)]

    def construct_solution_with_backtracking():
        """Build one full fold, backtracking out of dead ends up to 10 times."""
        temp_protein = LatticeProtein(hp_sequence, original_seq)
        backtrack_count = 0
        temp_protein._positions[0] = (0, 0, 0)
        step = 1
        while step < len(hp_sequence) and backtrack_count < 10:
            if step == 1:
                direction = random.choice(_DIRECTIONS)
                temp_protein._positions[step] = tuple(temp_protein._positions[0][i] + direction[i] for i in range(3))
                step += 1
            else:
                selected_direction = probabilistic_selection(temp_protein, step)
                if selected_direction is None:
                    backtrack_steps = min(3, step - 1)
                    step -= backtrack_steps
                    backtrack_count += 1
                    for i in range(step, len(temp_protein._positions)):
                        temp_protein._positions[i] = None
                    continue
                current_pos = temp_protein._positions[step - 1]
                temp_protein._positions[step] = tuple(current_pos[i] + selected_direction[i] for i in range(3))
                step += 1

        if step == len(hp_sequence):
            temp_protein._energy = temp_protein.calculate_energy()
            return temp_protein
        return None

    def construct_multiple_attempts():
        """Return the best of several construction attempts."""
        best_attempt = None
        attempt_best_energy = float("inf")
        for _ in range(5):
            solution = construct_solution_with_backtracking()
            if solution is not None and solution._energy < attempt_best_energy:
                attempt_best_energy = solution._energy
                best_attempt = solution.clone()
        return best_attempt

    stagnation_counter = 0
    last_best_energy = best_energy

    for iteration in range(max_iter):
        ants = [a for a in (construct_multiple_attempts() for _ in range(num_ants)) if a is not None]

        # If every ant failed, weaken pheromones and retry.
        if not ants:
            for step in pheromones:
                for direction in pheromones[step]:
                    pheromones[step][direction] *= 0.5
            continue

        for ant in ants:
            if ant._energy < best_energy:
                best_energy = ant._energy
                best_protein = ant.clone()
                best_positions = list(ant._positions)
                position_history.append(list(ant._positions))
                energy_history.append(best_energy)
                stagnation_counter = 0

        stagnation_counter = stagnation_counter + 1 if best_energy == last_best_energy else 0
        last_best_energy = best_energy

        # Reset pheromones if stuck for too long.
        if stagnation_counter > 50:
            for step in pheromones:
                for direction in pheromones[step]:
                    pheromones[step][direction] = initial_pheromone
            stagnation_counter = 0

        # Evaporate, then reinforce the better half of the ants.
        evap_rate = evaporation_rate * (1 + stagnation_counter * 0.01)
        for step in pheromones:
            for direction in pheromones[step]:
                pheromones[step][direction] *= (1 - min(evap_rate, 0.5))

        ants.sort(key=lambda x: x._energy)
        for rank, ant in enumerate(ants):
            if rank < num_ants // 2:
                pheromone_amount = (1.0 / (1.0 + abs(ant._energy))) * (num_ants - rank) / num_ants
                for step in range(1, min(sequence_length, len(ant._positions))):
                    if ant._positions[step] is not None and ant._positions[step - 1] is not None:
                        direction = tuple(ant._positions[step][i] - ant._positions[step - 1][i] for i in range(3))
                        if step in pheromones and direction in pheromones[step]:
                            pheromones[step][direction] += pheromone_amount

        if time.time() - program_starts >= time_limit:
            break

    if best_protein is not None:
        best_protein._positions = list(best_positions)
        best_protein._energy = best_energy

    return best_protein, energy_history, position_history
