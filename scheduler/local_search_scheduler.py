"""
Hill Climbing Local Search post-processor for ACO solutions.

Applies two neighbourhood operators to the best ACO solution:
  1. In-place channel swap  — replace a program with a same-duration alternative
                              on a different channel (no cascade effect on the tail).
  2. Greedy tail rebuild    — remove the suffix from the worst-fitness slot and
                              rebuild it with the greedy heuristic.

Only improving moves are accepted (strict hill climbing).
Stops when no operator finds an improvement or max_iterations is reached.
"""
from __future__ import annotations

import time as _time
from typing import Any, Dict, List, Optional, Tuple

from models.instance_data import InstanceData
from models.schedule import Schedule
from models.solution import Solution
from utils.algorithm_utils import AlgorithmUtils
from utils.scheduler_utils import SchedulerUtils
from utils.utils import Utils
from validator.validator import Validator

_DEFAULT_MAX_ITER    = 100   # hill-climbing iterations cap
_DEFAULT_MAX_REBUILD = 5     # worst-N slots tried per tail-rebuild pass


class LocalSearchScheduler:
    """
    Hill Climbing Local Search that post-processes a single solution.

    Parameters
    ----------
    instance_data   : problem instance (shared read-only)
    max_iterations  : outer loop cap (stops earlier if no improvement found)
    max_rebuilds    : how many worst-fitness slots to try for tail rebuild
                      per iteration
    verbose         : print per-iteration progress
    """

    def __init__(
        self,
        instance_data: InstanceData,
        max_iterations: int = _DEFAULT_MAX_ITER,
        max_rebuilds: int   = _DEFAULT_MAX_REBUILD,
        verbose: bool       = False,
    ) -> None:
        self.instance_data  = instance_data
        self.max_iterations = max_iterations
        self.max_rebuilds   = max_rebuilds
        self.verbose        = verbose

        # fast channel_id → list-index lookup
        self._ch_id_to_idx: Dict[int, int] = {
            ch.channel_id: idx
            for idx, ch in enumerate(instance_data.channels)
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def improve(
        self,
        initial_solution: Solution,
    ) -> Tuple[Solution, Dict[str, Any]]:
        """
        Apply Hill Climbing to *initial_solution*.

        Returns
        -------
        (improved_solution, stats_dict)

        stats_dict keys
        ---------------
        aco_score, ls_score, improvement, improvement_pct,
        iterations, swaps_applied, rebuilds_applied, ls_time_sec
        """
        scheduled: List[Schedule] = list(initial_solution.scheduled_programs)
        current_score             = initial_solution.total_score

        swaps_applied    = 0
        rebuilds_applied = 0
        iteration        = 0

        t_start = _time.perf_counter()

        for iteration in range(1, self.max_iterations + 1):
            improved = False

            # ── Operator 1: in-place channel swap ────────────────────
            for i in range(len(scheduled)):
                delta, new_i, new_i1 = self._best_swap(scheduled, i)
                if delta > 0:
                    scheduled[i] = new_i
                    if new_i1 is not None:
                        scheduled[i + 1] = new_i1
                    current_score += delta
                    swaps_applied += 1
                    improved = True

            # ── Operator 2: greedy tail rebuild (worst slots first) ───
            order = sorted(range(len(scheduled)), key=lambda k: scheduled[k].fitness)
            for i in order[: self.max_rebuilds]:
                old_tail_score = sum(s.fitness for s in scheduled[i:])
                new_tail, new_tail_score = self._greedy_tail(
                    scheduled[:i], scheduled[i].start
                )
                if new_tail_score > old_tail_score:
                    scheduled     = scheduled[:i] + new_tail
                    current_score = sum(s.fitness for s in scheduled)
                    rebuilds_applied += 1
                    improved = True
                    break   # restart both operators from scratch

            if not improved:
                break

        ls_time = _time.perf_counter() - t_start

        best_solution = Solution(scheduled, current_score)
        improvement   = current_score - initial_solution.total_score

        stats: Dict[str, Any] = {
            "aco_score":        initial_solution.total_score,
            "ls_score":         current_score,
            "improvement":      improvement,
            "improvement_pct":  round(
                improvement / max(initial_solution.total_score, 1) * 100, 2
            ),
            "iterations":       iteration,
            "swaps_applied":    swaps_applied,
            "rebuilds_applied": rebuilds_applied,
            "ls_time_sec":      round(ls_time, 3),
        }

        if self.verbose:
            sign = "+" if improvement >= 0 else ""
            print(
                f"  [LS] ACO={initial_solution.total_score} -> "
                f"LS={current_score} ({sign}{improvement}, "
                f"{sign}{stats['improvement_pct']}%)  "
                f"swaps={swaps_applied}  rebuilds={rebuilds_applied}  "
                f"iters={iteration}  time={stats['ls_time_sec']}s"
            )

        return best_solution, stats

    # ------------------------------------------------------------------
    # Operator 1: in-place channel swap (same start time, same end time)
    # ------------------------------------------------------------------

    def _best_swap(
        self,
        scheduled: List[Schedule],
        i: int,
    ) -> Tuple[int, Optional[Schedule], Optional[Schedule]]:
        """
        At position *i*, try every channel whose program starts AND ends at
        exactly the same times as the current program.  This guarantees no
        cascade effect on the tail.

        Also re-computes position i+1's fitness (switch penalty may change).

        Returns (best_delta, new_schedule_i, new_schedule_i1_or_None).
        A delta > 0 means a strict improvement was found.
        """
        sched_i  = scheduled[i]
        prefix   = scheduled[:i]
        t        = sched_i.start   # must be the program's real start time too

        best_delta  = 0
        best_new_i  = None
        best_new_i1: Optional[Schedule] = None

        for ch_idx, channel in enumerate(self.instance_data.channels):
            if channel.channel_id == sched_i.channel_id:
                continue   # same channel — nothing changes

            # ── constraint check at position i ───────────────────────
            if not Validator.is_channel_valid(
                prefix, self.instance_data, ch_idx, t
            ):
                continue

            program = Utils.get_channel_program_by_time(channel, t)
            if program is None:
                continue
            # Only accept programs that START at exactly time t
            # (never join mid-stream, consistent with ACO behaviour)
            if program.start != t:
                continue
            # Same end time → tail is unaffected
            if program.end != sched_i.end:
                continue
            if program.unique_id == sched_i.unique_program_id:
                continue
            # Do not re-schedule the same unique program that was just watched
            if prefix and prefix[-1].unique_program_id == program.unique_id:
                continue

            new_fit_i = int(self._fitness(prefix, channel, program, t))
            new_i = Schedule(
                program_id       = program.program_id,
                channel_id       = channel.channel_id,
                start            = program.start,
                end              = program.end,
                fitness          = new_fit_i,
                unique_program_id= program.unique_id,
            )

            # ── re-evaluate position i+1 (switch penalty may change) ─
            cand_new_i1: Optional[Schedule] = None
            if i + 1 < len(scheduled):
                sched_i1  = scheduled[i + 1]
                fake_pre  = prefix + [new_i]
                ch_idx_i1 = self._ch_id_to_idx.get(sched_i1.channel_id, -1)
                if ch_idx_i1 < 0:
                    continue
                if not Validator.is_channel_valid(
                    fake_pre, self.instance_data, ch_idx_i1, sched_i1.start
                ):
                    continue
                ch_i1   = self.instance_data.channels[ch_idx_i1]
                prog_i1 = Utils.get_program_by_unique_id(
                    self.instance_data, sched_i1.unique_program_id
                )
                if prog_i1 is None:
                    continue
                new_fit_i1 = int(
                    self._fitness(fake_pre, ch_i1, prog_i1, sched_i1.start)
                )
                cand_new_i1 = Schedule(
                    program_id       = sched_i1.program_id,
                    channel_id       = sched_i1.channel_id,
                    start            = sched_i1.start,
                    end              = sched_i1.end,
                    fitness          = new_fit_i1,
                    unique_program_id= sched_i1.unique_program_id,
                )
                delta = (new_fit_i + new_fit_i1) - (sched_i.fitness + sched_i1.fitness)
            else:
                delta = new_fit_i - sched_i.fitness

            if delta > best_delta:
                best_delta  = delta
                best_new_i  = new_i
                best_new_i1 = cand_new_i1

        return best_delta, best_new_i, best_new_i1

    # ------------------------------------------------------------------
    # Operator 2: greedy tail rebuild
    # ------------------------------------------------------------------

    def _greedy_tail(
        self,
        prefix: List[Schedule],
        start_time: int,
    ) -> Tuple[List[Schedule], int]:
        """
        Rebuild the schedule greedily from *start_time* with *prefix* fixed.
        Mirrors AntScheduler._build_greedy_solution exactly.
        """
        scheduled   = list(prefix)
        added: List[Schedule] = []
        added_score = 0
        cursor      = start_time

        while cursor < self.instance_data.closing_time:
            valid_idxs = SchedulerUtils.get_valid_schedules(
                scheduled, self.instance_data, cursor
            )
            if not valid_idxs:
                cursor += 1
                continue

            best_ch, best_prog, score = AlgorithmUtils.get_best_fit(
                scheduled, self.instance_data, cursor, valid_idxs
            )
            if best_ch is None or best_prog is None or score <= 0:
                cursor += 1
                continue
            if scheduled and best_prog.start < scheduled[-1].end:
                cursor += 1
                continue
            if scheduled and scheduled[-1].unique_program_id == best_prog.unique_id:
                cursor += 1
                continue

            s = Schedule(
                program_id       = best_prog.program_id,
                channel_id       = best_ch.channel_id,
                start            = best_prog.start,
                end              = best_prog.end,
                fitness          = score,
                unique_program_id= best_prog.unique_id,
            )
            scheduled.append(s)
            added.append(s)
            added_score += score
            cursor = best_prog.end

        return added, added_score

    # ------------------------------------------------------------------
    # Fitness helper (identical formula to AntScheduler._compute_heuristic)
    # ------------------------------------------------------------------

    def _fitness(
        self,
        prefix: List[Schedule],
        channel,
        program,
        schedule_time: int,
    ) -> float:
        score = float(program.score)
        score += AlgorithmUtils.get_time_preference_bonus(
            self.instance_data, program, schedule_time
        )
        score += AlgorithmUtils.get_switch_penalty(
            prefix, self.instance_data, channel
        )
        score += AlgorithmUtils.get_early_termination_penalty(
            prefix, self.instance_data, program, schedule_time
        )
        return max(score, 0.0)
