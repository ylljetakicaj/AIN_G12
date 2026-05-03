"""
ACO Parameter Fine-Tuning Experiment Runner
============================================
Runs the AntScheduler with multiple parameter configurations,
collecting up to 10 results per (instance x config) pair within a
5-minute time budget.  Each individual run is also capped at
max_run_sec seconds (default 600) via a subprocess timeout, so
extremely large instances never block indefinitely.

Usage
-----
    python experiment_runner.py                             # all instances
    python experiment_runner.py --instances germany kosovo  # subset
    python experiment_runner.py --runs 10 --time-limit 300 --max-run-sec 600
"""

import argparse
import csv
import io
import json
import multiprocessing as mp
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Force UTF-8 on Windows consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# =====================================================================
# PARAMETER CONFIGURATIONS
# =====================================================================

CONFIGS: List[Dict] = [
    {
        "name":        "C1_Baseline",
        "description": "Default ACO - balanced exploration/exploitation",
        "num_ants":        100,
        "num_generations":  10,
        "alpha":           2.0,
        "beta":            1.0,
        "rho":             0.3,
        "random_factor":   0.15,
    },
    {
        "name":        "C2_Exploitation",
        "description": "Exploitation-heavy: high alpha, slow evaporation, low randomness",
        "num_ants":         50,
        "num_generations":  20,
        "alpha":            3.0,
        "beta":             0.5,
        "rho":              0.2,
        "random_factor":    0.05,
    },
    {
        "name":        "C3_Exploration",
        "description": "Exploration-heavy: high beta, fast evaporation, high randomness",
        "num_ants":         80,
        "num_generations":  15,
        "alpha":            1.0,
        "beta":             2.0,
        "rho":              0.5,
        "random_factor":    0.30,
    },
    {
        "name":        "C4_Balanced",
        "description": "Balanced: alpha = beta, moderate evaporation and randomness",
        "num_ants":         60,
        "num_generations":  15,
        "alpha":            1.5,
        "beta":             1.5,
        "rho":              0.35,
        "random_factor":    0.10,
    },
    {
        "name":        "C5_WidePopulation",
        "description": "Wide population: many ants, few generations, moderate params",
        "num_ants":        150,
        "num_generations":   5,
        "alpha":            2.0,
        "beta":             1.5,
        "rho":              0.25,
        "random_factor":    0.20,
    },
]

NUM_RUNS        = 10
TIME_LIMIT_SECS = 300   # max seconds per (instance x config) pair
MAX_RUN_SECS    = 600   # max seconds for a single run (kills huge instances)


# =====================================================================
# SUBPROCESS WORKER  (must be module-level for Windows mp.spawn)
# =====================================================================

def _worker(queue: mp.Queue, input_path: str, cfg: dict, run_idx: int) -> None:
    """Runs one ACO execution in a child process and puts the score in queue."""
    # Re-import inside the worker — each subprocess is a fresh Python.
    sys.path.insert(0, str(Path(__file__).parent))
    from parser.parser import Parser
    from scheduler.ant_scheduler import AntScheduler
    from utils.utils import Utils

    try:
        instance_data = Parser(input_path).parse()
        Utils.set_current_instance(instance_data)
        scheduler = AntScheduler(
            instance_data   = instance_data,
            num_ants        = cfg["num_ants"],
            num_generations = cfg["num_generations"],
            alpha           = cfg["alpha"],
            beta            = cfg["beta"],
            rho             = cfg["rho"],
            random_factor   = cfg["random_factor"],
            random_seed     = run_idx * 13 + 7,
            verbose         = False,
        )
        score = scheduler.generate_solution().total_score
        queue.put(score)
    except Exception as exc:
        queue.put(None)


# =====================================================================
# DATA CLASSES
# =====================================================================

@dataclass
class RunRecord:
    instance:     str
    config_name:  str
    run:          int
    score:        int
    duration_sec: float
    timed_out:    bool


@dataclass
class ConfigSummary:
    instance:         str
    config_name:      str
    best:             int
    worst:            int
    avg:              float
    runs_completed:   int
    runs_timed_out:   int
    total_time_sec:   float


# =====================================================================
# EXPERIMENT RUNNER
# =====================================================================

class ExperimentRunner:
    """
    For every (instance, config) pair: runs ACO up to num_runs times.
    Stops the pair early when the cumulative budget (time_limit) is
    exhausted.  Each individual run is capped at max_run_sec via a
    subprocess; timed-out runs are recorded but excluded from stats.
    """

    def __init__(
        self,
        input_dir:   str = "data/input",
        output_dir:  str = "data/experiments",
        num_runs:    int = NUM_RUNS,
        time_limit:  int = TIME_LIMIT_SECS,
        max_run_sec: int = MAX_RUN_SECS,
        configs:     Optional[List[Dict]] = None,
        verbose:     bool = True,
    ) -> None:
        self.input_dir   = Path(input_dir)
        self.output_dir  = Path(output_dir)
        self.num_runs    = num_runs
        self.time_limit  = time_limit
        self.max_run_sec = max_run_sec
        self.configs     = configs if configs is not None else CONFIGS
        self.verbose     = verbose
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def run_all(self, instance_filter: Optional[List[str]] = None) -> None:
        instances = sorted(self.input_dir.glob("*.json"))
        if instance_filter:
            instances = [
                p for p in instances
                if any(f in p.stem for f in instance_filter)
            ]
        if not instances:
            print("No input files found. Check --input-dir or --instances.")
            return

        all_runs:  List[RunRecord]    = []
        summaries: List[ConfigSummary] = []

        for inst_path in instances:
            inst_runs, inst_summaries = self._run_instance(inst_path)
            all_runs.extend(inst_runs)
            summaries.extend(inst_summaries)

        self._save_csv(all_runs, summaries)
        self._save_json(all_runs, summaries)
        self._print_report(summaries)

    # ------------------------------------------------------------------
    def _run_instance(self, inst_path: Path):
        inst_name = inst_path.stem
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  INSTANCE: {inst_name}")
        print(f"{sep}")

        all_runs:  List[RunRecord]    = []
        summaries: List[ConfigSummary] = []

        for cfg in self.configs:
            cfg_name = cfg["name"]
            print(f"\n  Config: {cfg_name}  ({cfg['description']})")
            print(f"  alpha={cfg['alpha']}  beta={cfg['beta']}  rho={cfg['rho']}"
                  f"  ants={cfg['num_ants']}  gens={cfg['num_generations']}"
                  f"  rand={cfg['random_factor']}")
            print(f"  {'-' * 55}")

            cfg_scores:   List[int]   = []
            cfg_times:    List[float] = []
            cfg_timeouts: int         = 0
            budget_start = time.perf_counter()

            for run_idx in range(1, self.num_runs + 1):
                elapsed = time.perf_counter() - budget_start
                if elapsed >= self.time_limit:
                    print(f"  [!] Budget ({self.time_limit}s) reached"
                          f" - stopped before run {run_idx}")
                    break

                t0, score, timed_out = time.perf_counter(), None, False
                q = mp.Queue()
                p = mp.Process(
                    target=_worker,
                    args=(q, str(inst_path), cfg, run_idx),
                )
                p.start()
                p.join(timeout=self.max_run_sec)

                if p.is_alive():
                    p.terminate()
                    p.join()
                    timed_out = True
                else:
                    score = q.get() if not q.empty() else None

                dt = time.perf_counter() - t0

                if timed_out or score is None:
                    cfg_timeouts += 1
                    all_runs.append(RunRecord(
                        instance     = inst_name,
                        config_name  = cfg_name,
                        run          = run_idx,
                        score        = -1,
                        duration_sec = round(dt, 3),
                        timed_out    = True,
                    ))
                    if self.verbose:
                        total_elapsed = time.perf_counter() - budget_start
                        print(f"    Run {run_idx:2d}: TIMEOUT"
                              f"  time={dt:.1f}s"
                              f"  budget={total_elapsed:.1f}/{self.time_limit}s")
                else:
                    cfg_scores.append(score)
                    cfg_times.append(dt)
                    all_runs.append(RunRecord(
                        instance     = inst_name,
                        config_name  = cfg_name,
                        run          = run_idx,
                        score        = score,
                        duration_sec = round(dt, 3),
                        timed_out    = False,
                    ))
                    if self.verbose:
                        total_elapsed = time.perf_counter() - budget_start
                        print(f"    Run {run_idx:2d}: score={score:8,}"
                              f"  time={dt:.1f}s"
                              f"  budget={total_elapsed:.1f}/{self.time_limit}s")

            if cfg_scores:
                summaries.append(ConfigSummary(
                    instance       = inst_name,
                    config_name    = cfg_name,
                    best           = max(cfg_scores),
                    worst          = min(cfg_scores),
                    avg            = round(sum(cfg_scores) / len(cfg_scores), 1),
                    runs_completed = len(cfg_scores),
                    runs_timed_out = cfg_timeouts,
                    total_time_sec = round(sum(cfg_times), 2),
                ))
                print(f"  >> best={max(cfg_scores):,}"
                      f"  avg={sum(cfg_scores)/len(cfg_scores):,.1f}"
                      f"  worst={min(cfg_scores):,}"
                      f"  ({len(cfg_scores)} runs"
                      + (f", {cfg_timeouts} timeouts)" if cfg_timeouts else ")"))
            else:
                summaries.append(ConfigSummary(
                    instance       = inst_name,
                    config_name    = cfg_name,
                    best           = -1,
                    worst          = -1,
                    avg            = -1.0,
                    runs_completed = 0,
                    runs_timed_out = cfg_timeouts,
                    total_time_sec = 0.0,
                ))
                print(f"  >> All {cfg_timeouts} run(s) timed out - no results")

        return all_runs, summaries

    # ------------------------------------------------------------------
    def _save_csv(self, runs, summaries) -> None:
        runs_path = self.output_dir / "experiment_runs.csv"
        with open(runs_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["instance", "config", "run",
                        "score", "duration_sec", "timed_out"])
            for r in runs:
                w.writerow([r.instance, r.config_name, r.run,
                            r.score, r.duration_sec, r.timed_out])

        summ_path = self.output_dir / "experiment_summary.csv"
        with open(summ_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["instance", "config", "best", "avg", "worst",
                        "runs_completed", "runs_timed_out", "total_time_sec"])
            for s in summaries:
                w.writerow([s.instance, s.config_name,
                            s.best, s.avg, s.worst,
                            s.runs_completed, s.runs_timed_out,
                            s.total_time_sec])

        print(f"\n  Saved runs    -> {runs_path}")
        print(f"  Saved summary -> {summ_path}")

    def _save_json(self, runs, summaries) -> None:
        config_meta = {
            c["name"]: {k: v for k, v in c.items() if k != "name"}
            for c in self.configs
        }
        payload = {
            "meta": {
                "num_runs":       self.num_runs,
                "time_limit_sec": self.time_limit,
                "max_run_sec":    self.max_run_sec,
                "configs":        config_meta,
            },
            "runs":      [asdict(r) for r in runs],
            "summaries": [asdict(s) for s in summaries],
        }
        out_path = self.output_dir / "experiment_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"  Saved JSON    -> {out_path}")

    def _print_report(self, summaries) -> None:
        instances = sorted({s.instance for s in summaries})
        sep = "=" * 75
        print(f"\n\n{sep}")
        print("  FINAL RESULTS REPORT")
        print(sep)

        for inst in instances:
            rows = sorted(
                [s for s in summaries if s.instance == inst],
                key=lambda x: x.best,
                reverse=True,
            )
            print(f"\n  Instance: {inst}")
            print(f"  {'Config':<22} {'Best':>8} {'Avg':>10}"
                  f" {'Worst':>8} {'Runs':>5} {'TO':>4} {'Time(s)':>8}")
            print(f"  {'-' * 70}")
            for i, s in enumerate(rows):
                star = " *" if i == 0 else "  "
                best_s  = f"{s.best:,}"  if s.best  >= 0 else "N/A"
                avg_s   = f"{s.avg:,.1f}" if s.avg  >= 0 else "N/A"
                worst_s = f"{s.worst:,}" if s.worst >= 0 else "N/A"
                print(f"  {s.config_name:<22} {best_s:>8} {avg_s:>10}"
                      f" {worst_s:>8} {s.runs_completed:>5}"
                      f" {s.runs_timed_out:>4} {s.total_time_sec:>8.1f}{star}")

        if summaries:
            from collections import defaultdict
            config_totals: Dict[str, List[int]] = defaultdict(list)
            for s in summaries:
                if s.best >= 0:
                    config_totals[s.config_name].append(s.best)
            ranked = sorted(
                config_totals.items(),
                key=lambda kv: sum(kv[1]) / len(kv[1]),
                reverse=True,
            )
            print(f"\n  {'-' * 70}")
            print("  Overall ranking by average best-score across instances:")
            for rank, (name, bests) in enumerate(ranked, 1):
                avg = sum(bests) / len(bests)
                print(f"    {rank}. {name:<22}  avg-best = {avg:,.1f}"
                      f"  (over {len(bests)} instances)")
        print(f"\n{sep}\n")


# =====================================================================
# CLI
# =====================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ACO parameter fine-tuning experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir",   default="data/input",
                   help="Directory containing input .json files")
    p.add_argument("--output-dir",  default="data/experiments",
                   help="Directory where CSV/JSON results are written")
    p.add_argument("--instances",   nargs="*", metavar="NAME",
                   help="Restrict to instances whose filename contains NAME")
    p.add_argument("--runs",        type=int, default=NUM_RUNS,
                   help="Max runs per (instance x config) pair")
    p.add_argument("--time-limit",  type=int, default=TIME_LIMIT_SECS,
                   help="Max cumulative seconds per (instance x config) pair")
    p.add_argument("--max-run-sec", type=int, default=MAX_RUN_SECS,
                   help="Hard timeout per individual run (kills huge instances)")
    p.add_argument("--quiet",       action="store_true",
                   help="Suppress per-run output")
    return p.parse_args()


if __name__ == "__main__":
    # Required on Windows for multiprocessing.
    mp.freeze_support()
    args = _parse_args()
    runner = ExperimentRunner(
        input_dir   = args.input_dir,
        output_dir  = args.output_dir,
        num_runs    = args.runs,
        time_limit  = args.time_limit,
        max_run_sec = args.max_run_sec,
        verbose     = not args.quiet,
    )
    runner.run_all(instance_filter=args.instances)
