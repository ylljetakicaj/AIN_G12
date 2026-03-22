"""
SimulatedAnnealingScheduler  — Time-Boxed Hybrid (Greedy + GA)
===============================================================
Runs in a fixed wall-clock budget (default 8 seconds) regardless of
instance size.  Internally it is a population-based hybrid:

  Phase 1 — Population bootstrap (~12 % of budget)
    • 1 pure-greedy individual  (deterministic best start)
    • N sampled-greedy variants (channel sampling for diversity)
    • A few full-channel randomised-greedy variants

  Phase 2 — Time-boxed evolution (~88 % of budget)
    • 80 % crossover  (one-point / two-point / uniform)  — very fast, 0.1 ms
    • 20 % partial re-plan mutation                       — slower,  ~7 ms
    • Fresh-greedy injection when stalled > 40 iterations
    • Always keeps global best (elitism)

  Output — best individual found within the budget.

This is fundamentally different from both:
  • Beam Search  : forward tree search, gets stuck, slow on 500 + channels
  • Pure SA      : single trajectory, no population diversity
"""

import random
import time
from typing import Optional

from models.solution import Solution
from scheduler.genetic_scheduler import GeneticScheduler, Individual


class SimulatedAnnealingScheduler:

    def __init__(
        self,
        instance_data,
        time_budget:  float = 8.0,        # wall-clock seconds
        pop_size:     int   = 40,          # population cap
        random_seed:  Optional[int] = 42,
        verbose:      bool  = True,
        # legacy params kept for API compatibility (ignored)
        max_iter:     int   = 2000,
        start_temp:   float = 800.0,
        cooling_rate: float = 0.997,
        fast_mode:    bool  = True,
    ):
        self.instance_data = instance_data
        self.time_budget   = time_budget
        self.pop_size      = pop_size
        self.verbose       = verbose

        if random_seed is not None:
            random.seed(random_seed)

        # Reuse all GA infrastructure (preprocess, decode, greedy, crossover)
        self.helper = GeneticScheduler(instance_data, verbose=False)

    # ------------------------------------------------------------------
    # Internal helpers (thin wrappers around GA primitives)
    # ------------------------------------------------------------------

    def _crossover(self, a: Individual, b: Individual) -> Individual:
        r = random.random()
        n = self.helper.n_slots
        if r < 0.45:
            k = random.randrange(1, n)
            child = Individual(a.genes[:k] + b.genes[k:])
        elif r < 0.75:
            i, j = sorted(random.sample(range(n), 2))
            child = Individual(a.genes[:i] + b.genes[i:j] + a.genes[j:])
        else:
            child = Individual([
                a.genes[s] if random.random() < 0.5 else b.genes[s]
                for s in range(n)
            ])
        return self.helper._decode(child)

    def _mutate(self, parent: Individual) -> Individual:
        k  = random.randrange(self.helper.n_slots)
        ng = self.helper._greedy_genes(
            randomize=True, from_slot=k, prefix_genes=parent.genes
        )
        return self.helper._decode(Individual(ng))

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def generate_solution(self) -> Solution:
        n_ch   = self.helper.n_ch
        budget = self.time_budget
        t0     = time.time()

        if self.verbose:
            print(f"\n{'=' * 60}")
            print("TIME-BOXED HYBRID SCHEDULER")
            print(f"Channels: {n_ch} | Budget: {budget}s | Pop: {self.pop_size}")
            print(f"{'=' * 60}\n")

        # ----------------------------------------------------------
        # Phase 1 : population bootstrap
        # ----------------------------------------------------------
        pop_budget = max(0.3, min(2.0, budget * 0.12))
        sample     = max(10, n_ch // 10)

        # Seed with pure (deterministic) greedy — always the best start
        best_ind = self.helper._decode(
            Individual(self.helper._greedy_genes(randomize=False))
        )
        best = best_ind.clone()
        pop  = [best_ind.clone()]

        # Sampled-greedy variants until pop_budget elapsed
        while time.time() - t0 < pop_budget:
            ind = self.helper._decode(
                Individual(self.helper._greedy_genes(randomize=True,
                                                     sample_size=sample))
            )
            pop.append(ind)
            if ind.score > best.score:
                best = ind.clone()

        # A few full-channel randomised-greedy individuals
        for _ in range(min(5, self.pop_size - len(pop))):
            ind = self.helper._decode(
                Individual(self.helper._greedy_genes(randomize=True))
            )
            pop.append(ind)
            if ind.score > best.score:
                best = ind.clone()

        pop = sorted(pop, key=lambda x: -x.score)[:self.pop_size]

        if self.verbose:
            t_pop = time.time() - t0
            print(f"  Phase 1: {len(pop)} individuals, "
                  f"best={best.score}, t={t_pop:.2f}s")

        # ----------------------------------------------------------
        # Phase 2 : time-boxed evolution
        # ----------------------------------------------------------
        stall       = 0
        improvements = 0
        top_k       = min(10, len(pop))

        while time.time() - t0 < budget - 0.05:
            remaining = budget - (time.time() - t0)

            # --- choose operation ---
            if stall > 40:
                # stuck → inject a fresh greedy individual
                child = self.helper._decode(
                    Individual(self.helper._greedy_genes(randomize=True))
                )
                stall = 0

            elif remaining < 0.2 or random.random() < 0.80:
                # crossover  (fast: ~0.1 ms)
                a, b  = random.sample(pop[:top_k], 2)
                child = self._crossover(a, b)

            else:
                # mutation  (partial re-plan: ~7 ms)
                parent = pop[random.randrange(min(5, len(pop)))]
                child  = self._mutate(parent)

            # --- accept / update ---
            if child.score > best.score:
                best         = child.clone()
                improvements += 1
                stall        = 0
                pop.insert(0, best.clone())
                if self.verbose:
                    print(f"  ✓ New best: {best.score} "
                          f"(+{best.score - best_ind.score} vs greedy, "
                          f"t={time.time()-t0:.1f}s)")
            else:
                stall += 1
                pop.append(child)

            if len(pop) > self.pop_size:
                pop = sorted(pop, key=lambda x: -x.score)[:self.pop_size]

        # ----------------------------------------------------------
        # Result
        # ----------------------------------------------------------
        elapsed = time.time() - t0
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"FINAL: Score={best.score}, "
                  f"Programs={len(best.schedules)}, "
                  f"Time={elapsed:.1f}s, "
                  f"Improvements={improvements}")
            print(f"{'=' * 60}\n")

        return Solution(best.schedules, best.score)