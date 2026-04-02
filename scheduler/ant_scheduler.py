import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from models.instance_data import InstanceData
from models.schedule import Schedule
from models.solution import Solution
from utils.algorithm_utils import AlgorithmUtils
from utils.scheduler_utils import SchedulerUtils
from utils.utils import Utils

NUM_ANTS        = 10
NUM_GENERATIONS = 5
ALPHA           = 1.0
BETA            = 2.0
RHO             = 0.3
Q               = 100.0
RANDOM_FACTOR   = 0.15
TAU_INIT        = 1.0

W_TIME_PREF = 2.0
W_SAME_CH   = 1.5
W_CONSEC_OK = 1.3


@dataclass
class Slot:
    idx:   int
    start: int
    end:   int


@dataclass
class GenerationStats:
    generation:  int
    best_score:  int
    avg_score:   float
    worst_score: int


@dataclass
class AntResult:
    greedy_solution:    Solution
    best_ant_solution:  Solution
    generation_history: List[GenerationStats] = field(default_factory=list)


class AntScheduler:

    def __init__(
        self,
        instance_data:   InstanceData,
        num_ants:        int   = NUM_ANTS,
        num_generations: int   = NUM_GENERATIONS,
        alpha:           float = ALPHA,
        beta:            float = BETA,
        rho:             float = RHO,
        q:               float = Q,
        random_factor:   float = RANDOM_FACTOR,
        tau_init:        float = TAU_INIT,
        random_seed:     Optional[int] = 42,
        verbose:         bool  = True,
    ):
        self.instance_data   = instance_data
        self.num_ants        = num_ants
        self.num_generations = num_generations
        self.alpha           = alpha
        self.beta            = beta
        self.rho             = rho
        self.q               = q
        self.random_factor   = random_factor
        self.tau_init        = tau_init
        self.verbose         = verbose

        if random_seed is not None:
            random.seed(random_seed)

        self.slots: List[Slot] = self._build_slots()
        n_slots = len(self.slots)
        n_ch    = len(instance_data.channels)

        # ── TABELA FILLESTARE E FEROMONES: τ[slot][kanal] = TAU_INIT ─────────
        self._pheromone: List[List[float]] = [
            [tau_init] * n_ch for _ in range(n_slots)
        ]

        if self.verbose:
            print(f"\n[Init] Slots: {n_slots} (×{instance_data.min_duration} min)  "
                  f"| Kanale: {n_ch}  "
                  f"| Pheromone matrix: {n_slots}×{n_ch}  "
                  f"| τ₀={tau_init}")

    def _build_slots(self) -> List[Slot]:
        d = self.instance_data.min_duration
        slots, t, idx = [], self.instance_data.opening_time, 0
        while t + d <= self.instance_data.closing_time:
            slots.append(Slot(idx, t, t + d))
            t += d
            idx += 1
        return slots
    
    def generate_solution(self) -> Solution:
        """Wrapper për main.py — kthen zgjidhjen ACO më të mirë."""
        return self.run().best_ant_solution

    def run(self) -> AntResult:
        if self.verbose:
            print(f"\n{'=' * 65}")
            print("  ANT COLONY OPTIMIZATION — TV Scheduler")
            print(f"  Milingona: {self.num_ants}  |  Gjenerata: {self.num_generations}")
            print(f"  α={self.alpha}  β={self.beta}  ρ={self.rho}  "
                  f"RandomFactor={self.random_factor}")
            print(f"{'=' * 65}")

        greedy_sol = self._build_greedy_solution()
        if self.verbose:
            print(f"\n  [Zgjidhja 1 — Greedy]  "
                  f"Score={greedy_sol.total_score}  "
                  f"Programs={len(greedy_sol.scheduled_programs)}")

        best_solution = greedy_sol
        history: List[GenerationStats] = []

        for gen in range(1, self.num_generations + 1):

            population = self._generate_population()

            scores   = [s.total_score for s in population]
            gen_best = max(population, key=lambda s: s.total_score)
            stats    = GenerationStats(
                generation  = gen,
                best_score  = gen_best.total_score,
                avg_score   = round(sum(scores) / len(scores), 1),
                worst_score = min(scores),
            )
            history.append(stats)

            if gen_best.total_score > best_solution.total_score:
                best_solution = gen_best

            if self.verbose:
                marker = " ✓" if gen_best.total_score > greedy_sol.total_score else ""
                print(f"  [Gjenerata {gen:2d}]  "
                      f"Best={stats.best_score:6d}  "
                      f"Avg={stats.avg_score:8.1f}  "
                      f"Worst={stats.worst_score:6d}{marker}")

            # Përditëso feromonen
            self._update_pheromones(population)

        # ── KRAHASIMI FINAL ───────────────────────────────────────────────────
        if self.verbose:
            diff = best_solution.total_score - greedy_sol.total_score
            sign = "+" if diff >= 0 else ""
            print(f"\n{'=' * 65}")
            print(f"  Zgjidhja 1 (Greedy): {greedy_sol.total_score:8d}  "
                  f"({len(greedy_sol.scheduled_programs)} programe)")
            print(f"  Zgjidhja 2 (ACO):    {best_solution.total_score:8d}  "
                  f"({len(best_solution.scheduled_programs)} programe)  "
                  f"[{sign}{diff} ndaj Greedy]")
            print(f"{'=' * 65}\n")

        return AntResult(
            greedy_solution    = greedy_sol,
            best_ant_solution  = best_solution,
            generation_history = history,
        )

    def _build_greedy_solution(self) -> Solution:
        """Zgjidhja 1 — plotësisht greedy, pa randomness, pa feromona."""
        time, total_score, scheduled = self.instance_data.opening_time, 0, []

        while time < self.instance_data.closing_time:
            valid_idxs = SchedulerUtils.get_valid_schedules(
                scheduled, self.instance_data, time
            )
            if not valid_idxs:
                time += 1
                continue

            best_ch, best_prog, score = AlgorithmUtils.get_best_fit(
                scheduled, self.instance_data, time, valid_idxs
            )
            if best_ch is None or best_prog is None or score <= 0:
                time += 1
                continue
            if scheduled and best_prog.start < scheduled[-1].end:
                time += 1
                continue
            if scheduled and scheduled[-1].unique_program_id == best_prog.unique_id:
                time += 1
                continue

            scheduled.append(Schedule(
                program_id=best_prog.program_id, channel_id=best_ch.channel_id,
                start=best_prog.start, end=best_prog.end,
                fitness=score, unique_program_id=best_prog.unique_id,
            ))
            time = best_prog.end
            total_score += score

        return Solution(scheduled, total_score)

    def _generate_population(self) -> List[Solution]:
        return [self._ant_build_solution(i) for i in range(self.num_ants)]

    def _ant_build_solution(self, ant_id: int) -> Solution:
        time, total_score, scheduled = self.instance_data.opening_time, 0, []

        while time < self.instance_data.closing_time:
            slot_idx   = self._time_to_slot_idx(time)
            valid_idxs = SchedulerUtils.get_valid_schedules(
                scheduled, self.instance_data, time
            )
            if not valid_idxs:
                time += self.instance_data.min_duration
                continue

            candidates = self._collect_candidates(
                scheduled, valid_idxs, time, slot_idx
            )
            if not candidates:
                time += self.instance_data.min_duration
                continue

            best_ch, best_prog = self._select_candidate(candidates, slot_idx)
            fitness = int(self._compute_heuristic(
                scheduled, best_ch, best_prog, time
            ))

            scheduled.append(Schedule(
                program_id=best_prog.program_id, channel_id=best_ch.channel_id,
                start=best_prog.start, end=best_prog.end,
                fitness=fitness, unique_program_id=best_prog.unique_id,
            ))
            time        = best_prog.end
            total_score += fitness

        return Solution(scheduled, total_score)

    def _collect_candidates(
        self,
        scheduled:  List[Schedule],
        valid_idxs: List[int],
        time:       int,
        slot_idx:   int,
    ) -> List[Tuple]:
        candidates = []
        for idx in valid_idxs:
            channel = self.instance_data.channels[idx]
            program = Utils.get_channel_program_by_time(channel, time)
            if program is None:
                continue
            if program.end - program.start < self.instance_data.min_duration:
                continue
            if scheduled and program.start < scheduled[-1].end:
                continue
            if scheduled and scheduled[-1].unique_program_id == program.unique_id:
                continue
            heur = self._compute_heuristic(scheduled, channel, program, time)
            if heur > 0:
                candidates.append((channel, program, heur, idx))
        return candidates

    def _compute_heuristic(self, scheduled, channel, program, time: int) -> float:
        score = float(program.score)
        score += AlgorithmUtils.get_time_preference_bonus(
            self.instance_data, program, time
        )
        score += AlgorithmUtils.get_switch_penalty(
            scheduled, self.instance_data, channel
        )
        score += AlgorithmUtils.get_early_termination_penalty(
            scheduled, self.instance_data, program, time
        )
        return max(score, 0.01)

    def _select_candidate(self, candidates: List[Tuple], slot_idx: int) -> Tuple:

        if random.random() < self.random_factor:
            ch, prog, _, _ = random.choice(candidates)
            return ch, prog

        weights = []
        for channel, program, heur, ch_idx in candidates:
            tau = self._pheromone[slot_idx][ch_idx]
            weights.append((tau ** self.alpha) * (heur ** self.beta))

        total = sum(weights)
        if total <= 0:
            ch, prog, _, _ = random.choice(candidates)
            return ch, prog

        r = random.random() * total
        cum = 0.0
        for i, (ch, prog, _, _) in enumerate(candidates):
            cum += weights[i]
            if r <= cum:
                return ch, prog

        ch, prog, _, _ = candidates[-1]
        return ch, prog

    def _update_pheromones(self, population: List[Solution]) -> None:

        n_ch = len(self.instance_data.channels)

        for s in range(len(self.slots)):
            for c in range(n_ch):
                self._pheromone[s][c] = max(
                    self._pheromone[s][c] * (1.0 - self.rho), 0.001
                )

        for solution in population:
            if solution.total_score <= 0:
                continue
            base = self.q / solution.total_score

            for i, sched in enumerate(solution.scheduled_programs):
                slot_idx = self._time_to_slot_idx(sched.start)
                if not (0 <= slot_idx < len(self.slots)):
                    continue
                ch_idx = self._channel_id_to_idx(sched.channel_id)
                if ch_idx < 0:
                    continue

                prog = Utils.get_program_by_unique_id(
                    self.instance_data, sched.unique_program_id
                )

                mult = 1.0

                if prog and self._satisfies_time_preference(prog):
                    mult *= W_TIME_PREF         

                if i > 0:
                    prev = solution.scheduled_programs[i - 1]
                    if prev.channel_id == sched.channel_id:
                        mult *= W_SAME_CH       

                if self._consecutive_genre_ok(
                    solution.scheduled_programs[:i], prog
                ):
                    mult *= W_CONSEC_OK         

                self._pheromone[slot_idx][ch_idx] += base * mult

    def _time_to_slot_idx(self, time: int) -> int:
        d   = self.instance_data.min_duration
        idx = (time - self.instance_data.opening_time) // d
        return max(0, min(idx, len(self.slots) - 1))

    def _channel_id_to_idx(self, channel_id: int) -> int:
        for i, ch in enumerate(self.instance_data.channels):
            if ch.channel_id == channel_id:
                return i
        return -1

    def _satisfies_time_preference(self, program) -> bool:
        for pref in self.instance_data.time_preferences:
            if program.genre == pref.preferred_genre:
                overlap = min(program.end, pref.end) - max(program.start, pref.start)
                if overlap >= self.instance_data.min_duration:
                    return True
        return False

    def _consecutive_genre_ok(
        self, prev_scheduled: List[Schedule], program
    ) -> bool:
        if program is None or not prev_scheduled:
            return True
        count = 0
        for sched in reversed(prev_scheduled):
            p = Utils.get_program_by_unique_id(
                self.instance_data, sched.unique_program_id
            )
            if p is None or p.genre != program.genre:
                break
            count += 1
        return (count + 1) <= self.instance_data.max_consecutive_genre