"""
ga_node_search.py

Genetic algorithm to search for small sets of nodes whose removal reconnects a set of subject graphs.

Usage (from notebook):
    from ga_node_search import run_ga, load_graphs
    broken_subject_graphs = load_graphs(subjects_over_max, folder='networkx_graphs')
    best = run_ga(broken_subject_graphs, candidate_nodes=disc_node_set, pop_size=100, gens=200, max_removed=10)

The module exposes:
- run_ga(broken_subject_graphs, candidate_nodes, ...)
- load_graphs(subject_list, folder='networkx_graphs')

The GA uses binary encoding over the candidate node list (1=remove node). Fitness = subjects_fixed - penalty * n_removed.
"""

import random
import time
from typing import Dict, List, Tuple, Iterable, Set
import pickle
import os
import networkx as nx
import numpy as np

# ------------------------- Utils -------------------------

def load_graphs(subjects: Iterable[str], folder: str = 'networkx_graphs') -> Dict[str, nx.Graph]:
    """Load pickled networkx graphs for given subject ids.

    Expect files named like '<subject>_graph.gpickle' in `folder`
    Returns dict subject -> graph
    """
    graphs = {}
    for subj in subjects:
        fname = os.path.join(folder, f"{subj}_graph.gpickle")
        if not os.path.exists(fname):
            # try without suffix if user provided exact filename
            alt = os.path.join(folder, subj)
            if os.path.exists(alt):
                fname = alt
            else:
                raise FileNotFoundError(f"Graph file not found for subject {subj}: {fname}")
        with open(fname, 'rb') as f:
            graphs[subj] = pickle.load(f)
    return graphs


# ------------------------- GA evaluation -------------------------

class Evaluator:
    """Evaluator for subsets of nodes to remove from subject graphs.
    Caches results for efficiency.
    """
    def __init__(self, broken_subject_graphs: Dict[str, nx.Graph]):
        self.graphs = broken_subject_graphs
        # cache results for evaluated frozenset of nodes
        self.cache = {}

    def evaluate_subset(self, nodes_to_remove: Iterable[int]) -> Tuple[int, int]:
        """Evaluate how many subjects become connected when removing the given nodes.

        Returns (n_fixed, n_subjects_tested).
        Caches results keyed by frozenset(nodes_to_remove).
        """
        key = frozenset(nodes_to_remove) # use frozenset for cache key - immutable and order-independent
        if key in self.cache:
            return self.cache[key]

        n_fixed = 0
        n_subj = 0
        nodes_to_remove_set = set(nodes_to_remove)
        for subj, g in self.graphs.items():
            gcopy = g.copy()
            # remove (if present)
            for node in nodes_to_remove_set:
                if node in gcopy:
                    gcopy.remove_node(node)
            # After removal, check connectedness
            # If graph has <2 nodes, treat as not fixed
            if gcopy.number_of_nodes() >= 1 and nx.is_connected(gcopy):
                n_fixed += 1
            n_subj += 1

        self.cache[key] = (n_fixed, n_subj)
        return self.cache[key]


# ------------------------- GA implementation -------------------------

def individual_random(n_candidates: int, max_removed: int) -> np.ndarray:
    """Create a random binary individual with at most `max_removed` ones."""
    n_ones = random.randint(1, max_removed) if max_removed > 0 else 0
    ind = np.zeros(n_candidates, dtype=np.int8)
    if n_ones > 0:
        ones_idx = random.sample(range(n_candidates), k=n_ones)
        ind[ones_idx] = 1
    return ind


def repair_individual(ind: np.ndarray, max_removed: int) -> None:
    """In-place repair so that number of ones <= max_removed by randomly turning off excess ones."""
    ones = np.flatnonzero(ind)
    if len(ones) > max_removed:
        to_off = random.sample(list(ones), k=(len(ones) - max_removed))
        ind[to_off] = 0


def mutate(ind: np.ndarray, p_mut: float, max_removed: int) -> None:
    """Bitflip mutation in-place followed by repair."""
    for i in range(len(ind)):
        if random.random() < p_mut:
            ind[i] = 1 - ind[i]
    repair_individual(ind, max_removed)


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform crossover returning two children."""
    n = len(parent1)
    mask = np.random.rand(n) < 0.5
    child1 = parent1.copy()
    child2 = parent2.copy()
    child1[mask] = parent2[mask]
    child2[mask] = parent1[mask]
    return child1, child2


def tournament_selection(pop: List[np.ndarray], fitnesses: List[float], k: int = 3) -> np.ndarray:
    idxs = random.sample(range(len(pop)), k)
    best = idxs[0]
    for i in idxs[1:]:
        if fitnesses[i] > fitnesses[best]:
            best = i
    return pop[best].copy()


def decode_individual(ind: np.ndarray, candidates: List[int]) -> List[int]:
    return [candidates[i] for i, bit in enumerate(ind) if bit == 1]


def fitness_function(n_fixed: int, n_subj: int, n_removed: int, penalty: float = 0.3) -> float:
    """Compute fitness: prioritize n_fixed (subjects fixed), penalize larger sets.

    penalty is per node removed and is subtracted from fraction-fixed.
    We return a float where higher is better.
    """
    if n_subj == 0:
        return 0.0
    frac_fixed = n_fixed / n_subj
    return frac_fixed - penalty * (n_removed / max(1, n_subj))


def run_ga(
    broken_subject_graphs: Dict[str, nx.Graph],
    candidate_nodes: List[int],
    pop_size: int = 200,
    gens: int = 300,
    max_removed: int = 10,
    p_mut: float = 0.02,
    elitism: int = 2,
    penalty: float = 0.3,
    tournament_k: int = 3,
    seed: int = None,
    verbose: bool = True,
):
    """Main GA runner!

    Returns a dict with best individual, fitness history, and top-k found subsets.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    candidates = list(candidate_nodes)
    n_candidates = len(candidates)
    if n_candidates == 0:
        raise ValueError("No candidate nodes provided")

    evaluator = Evaluator(broken_subject_graphs)
    population = [individual_random(n_candidates, max_removed) for _ in range(pop_size)]

    # Precompute fitnesses
    fitnesses = []
    for ind in population:
        subset = decode_individual(ind, candidates)
        n_fixed, n_subj = evaluator.evaluate_subset(subset)
        fit = fitness_function(n_fixed, n_subj, len(subset), penalty)
        fitnesses.append(fit)

    best_overall = None
    best_fit = -1e9
    fitness_history = []
    start_time = time.time()

    for gen in range(1, gens + 1):
        new_pop = []
        new_fit = []

        # keep top `elitism` individuals
        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        for i in ranked[:elitism]:
            new_pop.append(population[i].copy())
            new_fit.append(fitnesses[i])

        # Fill rest of new_pop
        while len(new_pop) < pop_size:
            parent1 = tournament_selection(population, fitnesses, tournament_k)
            parent2 = tournament_selection(population, fitnesses, tournament_k)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1, p_mut, max_removed)
            mutate(child2, p_mut, max_removed)

            # evaluate children
            for child in (child1, child2):
                if len(new_pop) >= pop_size:
                    break
                subset = decode_individual(child, candidates)
                n_fixed, n_subj = evaluator.evaluate_subset(subset)
                fit = fitness_function(n_fixed, n_subj, len(subset), penalty)
                new_pop.append(child)
                new_fit.append(fit)

        population = new_pop
        fitnesses = new_fit

        # update best
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fit = fitnesses[gen_best_idx]
        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_overall = population[gen_best_idx].copy()

        fitness_history.append(best_fit)

        if verbose and (gen % max(1, gens // 10) == 0 or gen == 1):
            elapsed = time.time() - start_time
            print(f"Gen {gen}/{gens}  best_fit={best_fit:.4f}  elapsed={elapsed:.1f}s")

    # Build top-k results from cache
    results = []
    # Evaluate unique individuals in final population
    seen = set()
    for ind, fit in zip(population, fitnesses):
        subset = tuple(sorted(decode_individual(ind, candidates)))
        if subset in seen:
            continue
        seen.add(subset)
        n_fixed, n_subj = evaluator.evaluate_subset(subset)
        results.append({'subset': subset, 'n_fixed': n_fixed, 'n_subjects': n_subj, 'n_removed': len(subset), 'fitness': fit})

    results = sorted(results, key=lambda x: (x['fitness'], x['n_fixed']), reverse=True)

    best_subset = tuple(sorted(decode_individual(best_overall, candidates)))
    best_n_fixed, best_n_subj = evaluator.evaluate_subset(best_subset)

    return {
        'best_subset': best_subset,
        'best_n_fixed': best_n_fixed,
        'best_n_subjects': best_n_subj,
        'best_n_removed': len(best_subset),
        'best_fitness': best_fit,
        'fitness_history': fitness_history,
        'top_results': results[:20],
        'evaluator_cache_size': len(evaluator.cache),
    }


# ------------------------- main -------------------------
if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Run GA to search node-removals.')
    parser.add_argument('--subjects-file', type=str, default=None,
                        help='Optional newline-separated file with subject ids to load from networkx_graphs')
    parser.add_argument('--candidates', type=str, default=None,
                        help='Optional comma-separated list of candidate node indices (e.g. 12,23,45)')
    parser.add_argument('--pop-size', type=int, default=200)
    parser.add_argument('--gens', type=int, default=300)
    parser.add_argument('--max-removed', type=int, default=10)
    args = parser.parse_args()

    if args.subjects_file is None or args.candidates is None:
        print('Example usage: python ga_node_search.py --subjects-file subjects.txt --candidates 5,12,34,56')
        raise SystemExit(0)

    with open(args.subjects_file, 'r') as f:
        subjects = [l.strip() for l in f if l.strip()]
    candidates = [int(x) for x in args.candidates.split(',') if x.strip()]

    graphs = load_graphs(subjects)
    out = run_ga(graphs, candidates, pop_size=args.pop_size, gens=args.gens, max_removed=args.max_removed)
    print(json.dumps(out, indent=2, default=lambda o: list(o) if hasattr(o, '__iter__') else str(o)))
