<table border="0">
 <tr>
    <td><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/University_of_Prishtina_logo.svg/250px-University_of_Prishtina_logo.svg.png" width="150" alt="University Logo" /></td>
    <td>
      <p>University of Prishtina</p>
      <p>Faculty of Electrical and Computer Engineering</p>
      <p>Computer and Software Engineering — Master's Programme</p>
      <p>Professor: Prof. Dr. Kadri Sylejmani</p>
      <p>Assistant: MSc. Labeat Arbneshi</p>
    </td>
 </tr>
</table>

---

## Project Description: TV Scheduling Optimization

This project addresses the **Smart TV Schedule Optimizer** within the **Nature inspired algorithms** course. The primary objective is the optimal selection and scheduling of a subset of television programs across multiple channels, with the goal of maximizing total viewer score points.

### Constraints and Objectives

| Constraint | Description |
|---|---|
| **Time Window** | Programs must be scheduled strictly within the global opening/closing time interval |
| **No Overlap** | Simultaneous programs on the same channel are strictly prohibited |
| **Minimum Duration** | Programs must meet a minimum duration threshold to be considered valid |
| **Genre Repetition** | A limit on the number of consecutive programs of the same genre ensures variety |
| **Priority Blocks** | Specific time windows where only certain channels are allowed to broadcast |
| **Time Preferences** | Bonus points for broadcasting specific genres during preferred time slots |
| **Switch Penalty** | Points deducted when switching between channels |
| **Termination Penalty** | Points deducted when a program is cut off early |

**Optimization goal:** Maximize `total_score = sum(program scores) + time preference bonuses - switch penalties - termination penalties`

---

## Algorithms

### Beam Search Scheduler

A **deterministic** algorithm that overcomes the limitations of standard greedy approaches through parallel exploration of the solution space.

**Methodology:**
1. **Beam Search Strategy:** Instead of following a single path, the algorithm maintains a set of N most promising partial solutions at each step (**Beam Width**). This avoids local minima and recovers from sub-optimal early decisions.
2. **Lookahead Mechanism:** Beyond immediate evaluation, a configurable-depth lookahead analyses the impact of current decisions on future opportunities, preventing high-value programs from being blocked.
3. **Density Heuristic:** Evaluates remaining time slots using a score-per-minute density metric.

**Parameters:** Beam Width = 100 | Lookahead = 4 steps | Density Percentile = 25%

---

### Ant Colony Optimization (ACO) Scheduler

A **metaheuristic** algorithm that builds probabilistic solutions guided by pheromone trails and a local heuristic.

**How it works:**
- The scheduler is divided into fixed-length time **slots** (of `min_duration` minutes).
- A pheromone matrix `τ[slot][channel]` is maintained across generations.
- Each ant builds a complete schedule by choosing at each slot a (channel, program) pair with probability proportional to `τ^α × η^β`, where `η` is the heuristic score.
- After each generation, pheromones evaporate at rate `ρ` and are reinforced by the quality of each solution.
- A pure greedy solution is always generated as a baseline for comparison.

**Key parameters:**

| Parameter | Role |
|---|---|
| `num_ants` | Population size — more ants = broader coverage per generation |
| `num_generations` | Number of iterations — more generations = more learning |
| `alpha (α)` | Pheromone weight — high value → strong exploitation of known good paths |
| `beta (β)` | Heuristic weight — high value → decisions driven by local quality |
| `rho (ρ)` | Evaporation rate — high value → pheromone fades fast → more exploration |
| `random_factor` | Probability of a fully random jump → diversity and diversification |

---

## Running the Project

```bash
python main.py

# Specify algorithm directly
python main.py --algo ant    # Ant Colony Optimization (default)
python main.py --algo beam   # Beam Search
```

Output solutions are saved to `data/output/`.

---

## Parameter Fine-Tuning Experiments

### Methodology

To evaluate the impact of ACO parameters, **5 configurations** were tested across **13 feasible instances**. Each configuration was run **up to 10 times per instance** with a maximum time budget of **5 minutes per (instance × configuration) pair**. For large instances where a single run exceeds the budget, fewer runs were completed (noted in the tables).

Results are saved automatically by `experiment_runner.py`:

```
data/experiments/
├── experiment_runs.csv       # Every individual run: instance, config, run, score, duration
├── experiment_summary.csv    # Summary: best / avg / worst per (instance × config)
└── experiment_results.json   # Full JSON with all metadata and results
```

**To reproduce experiments:**
```bash
python experiment_runner.py                                    # all instances
python experiment_runner.py --instances germany kosovo croatia # specific instances
python experiment_runner.py --runs 10 --time-limit 300        # custom parameters
```

---

### Parameter Configurations

| # | Name | num_ants | num_gens | α | β | ρ | rand_factor | Strategy |
|---|------|:---:|:---:|:---:|:---:|:---:|:---:|---|
| C1 | **C1_Baseline** | 100 | 10 | 2.0 | 1.0 | 0.30 | 0.15 | Balanced default — reference point |
| C2 | **C2_Exploitation** | 50 | 20 | 3.0 | 0.5 | 0.20 | 0.05 | Heavy pheromone trust, slow forgetting |
| C3 | **C3_Exploration** | 80 | 15 | 1.0 | 2.0 | 0.50 | 0.30 | Heuristic-driven, fast evaporation |
| C4 | **C4_Balanced** | 60 | 15 | 1.5 | 1.5 | 0.35 | 0.10 | Equal α=β, moderate evaporation |
| C5 | **C5_WidePopulation** | 150 | 5 | 2.0 | 1.5 | 0.25 | 0.20 | Large population, few generations |

**Configuration rationale:**

- **C1_Baseline** — The project's default settings. Serves as a reproducible reference.
- **C2_Exploitation** — `α=3.0` gives strong weight to pheromones (ants follow known good trails). `ρ=0.2` means pheromone evaporates slowly — memory is preserved long. `random_factor=0.05` minimises random jumps. Risk: may converge prematurely.
- **C3_Exploration** — `β=2.0` means local heuristic quality dominates over pheromone. `ρ=0.5` erases trails quickly, forcing the algorithm to re-explore. High `random_factor=0.30` adds diversity. Risk: slow convergence.
- **C4_Balanced** — `α=β=1.5` ensures neither pheromone nor heuristic dominates. Moderate parameters aim for broad generalisability across instance sizes.
- **C5_WidePopulation** — 150 ants per generation provides much wider solution coverage each round at the cost of fewer generations (5). Well-suited when the time budget is tight, as fewer generations are needed to produce competitive results.

---

## Experimental Results

> **Note on run counts:** Instances with short runtimes (toy, germany, netherlands, kosovo, croatia) completed all 10 runs. Medium instances (spain, france, uk_tv_input) completed 5–9 runs. Large instances (singapore, australia, canada, usa_tv_input) completed 1–4 runs per config. For `uk_iptv`, configurations C1–C3 timed out on their first run (each run exceeds 5 min); only C4 and C5 produced results.

---

### Instance: `toy`

| Configuration | Best | Avg | Worst | Runs | Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|
| C1_Baseline | 380 | 380.0 | 380 | 10 | 7.2 |
| C2_Exploitation | 380 | 380.0 | 380 | 10 | 6.6 |
| C3_Exploration | 380 | 380.0 | 380 | 10 | 8.7 |
| C4_Balanced | 380 | 380.0 | 380 | 10 | 6.4 |
| C5_WidePopulation | 380 | 380.0 | 380 | 10 | 5.4 |

**Result:** All configurations converge to the same optimal score (380) with zero variance — the toy instance is trivially solved by ACO.

---

### Instance: `germany_tv_input`

| Configuration | Best | Avg | Worst | Runs | Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|
| C1_Baseline | 932 | 919.9 | 912 | 10 | 10.8 |
| C2_Exploitation | 922 | 915.3 | 902 | 10 | 10.5 |
| **C3_Exploration** | **932** | **925.6** | **917** | 10 | 12.2 |
| C4_Balanced | 923 | 918.8 | 900 | 10 | 10.4 |
| C5_WidePopulation | 932 | 918.6 | 910 | 10 | 8.1 |

**Best configuration:** C3_Exploration — highest average (925.6) and tied best score (932)

---

### Instance: `netherlands_tv_input`

| Configuration | Best | Avg | Worst | Runs | Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|
| C1_Baseline | 1,555 | 1,496.6 | 1,403 | 10 | 32.0 |
| C2_Exploitation | 1,559 | 1,430.3 | 1,308 | 10 | 31.4 |
| C3_Exploration | 1,598 | 1,501.6 | 1,449 | 10 | 40.3 |
| **C4_Balanced** | **1,664** | 1,495.2 | 1,417 | 10 | 29.7 |
| C5_WidePopulation | 1,531 | **1,490.1** | 1,424 | 10 | 25.6 |

**Best configuration:** C4_Balanced — highest best score (1,664); C3_Exploration leads on average (1,501.6)

---

### Instance: `kosovo_tv_input`

| Configuration | Best | Avg | Worst | Runs | Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|
| C1_Baseline | 1,467 | 1,425.8 | 1,384 | 10 | 31.6 |
| C2_Exploitation | 1,387 | 1,355.1 | 1,293 | 10 | 32.4 |
| **C3_Exploration** | **1,505** | **1,437.9** | **1,397** | 10 | 38.3 |
| C4_Balanced | 1,462 | 1,423.2 | 1,384 | 10 | 30.1 |
| C5_WidePopulation | 1,490 | 1,439.1 | 1,386 | 10 | 24.0 |

**Best configuration:** C3_Exploration — highest best score (1,505) and highest average (1,437.9)

---

### Instance: `croatia_tv_input`

| Configuration | Best | Avg | Worst | Runs | Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|
| C1_Baseline | 1,703 | 1,638.8 | 1,591 | 10 | 26.7 |
| C2_Exploitation | 1,705 | 1,600.4 | 1,482 | 10 | 25.2 |
| C3_Exploration | 1,867 | **1,768.3** | 1,642 | 10 | 34.5 |
| C4_Balanced | 1,819 | 1,680.8 | 1,598 | 10 | 22.8 |
| **C5_WidePopulation** | **1,974** | 1,751.2 | 1,635 | 10 | 21.3 |

**Best configuration:** C5_WidePopulation — highest best score (1,974); C3_Exploration leads on average (1,768.3)

---

### Instance: `spain_iptv`

| Configuration | Best | Avg | Worst | Runs | Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|
| C1_Baseline | 2,172 | 2,096.7 | 1,987 | 6 | 304.1 |
| **C2_Exploitation** | **2,361** | **2,103.3** | 2,012 | 7 | 329.0 |
| C3_Exploration | 2,112 | 2,068.0 | 2,006 | 6 | 305.1 |
| C4_Balanced | 2,199 | 2,060.7 | **1,935** | 9 | 328.7 |
| C5_WidePopulation | 2,252 | 2,060.0 | 1,965 | 4 | 765.2 |

**Best configuration:** C2_Exploitation — highest best score (2,361) and highest average (2,103.3)

---

### Instance: `france_iptv`

| Configuration | Best | Avg | Worst | Runs | Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|
| C1_Baseline | 2,029 | 1,979.3 | 1,913 | 6 | 351.8 |
| C2_Exploitation | 2,047 | 1,952.7 | 1,864 | 6 | 323.0 |
| C3_Exploration | 2,069 | **2,011.4** | 1,913 | 5 | 316.8 |
| C4_Balanced | 2,044 | 1,985.4 | **1,892** | 7 | 329.4 |
| **C5_WidePopulation** | **2,094** | 2,004.0 | 1,940 | 5 | 2429.7 |

**Best configuration:** C5_WidePopulation — highest best score (2,094); C3_Exploration leads on average (2,011.4)

---

### Instance: `singapore_pw`

| Configuration | Best | Avg | Worst | Runs | Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|
| C1_Baseline | 2,516 | 2,433.5 | 2,371 | 4 | 349.9 |
| C2_Exploitation | 2,596 | 2,475.0 | 2,320 | 4 | 304.9 |
| C3_Exploration | 2,545 | 2,458.2 | 2,364 | 4 | 362.8 |
| **C4_Balanced** | **2,642** | **2,422.6** | 2,299 | 5 | 330.7 |
| C5_WidePopulation | 2,442 | 2,442.0 | 2,442 | 1 | 599.9 |

**Best configuration:** C4_Balanced — highest best score (2,642)

---

### Instance: `uk_tv_input` — 120 channels

| Configuration | Best | Avg | Worst | Runs | Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|
| C1_Baseline | 1,538 | 1,431.2 | 1,341 | 10 | 253.0 |
| **C2_Exploitation** | **1,563** | 1,369.3 | 1,247 | 10 | 278.4 |
| C3_Exploration | 1,521 | 1,459.8 | **1,414** | 10 | 317.7 |
| **C4_Balanced** | 1,540 | **1,466.4** | 1,377 | 10 | 193.7 |
| C5_WidePopulation | 1,533 | 1,436.9 | 1,348 | 10 | 242.3 |

**Best configuration:** C2_Exploitation — highest best score (1,563); C4_Balanced — highest average (1,466.4) and fastest runtime (193.7s)

---

### Instance: `australia_iptv` — time-limited (2–3 runs per config)

| Configuration | Best | Avg | Worst | Runs | Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|
| C1_Baseline | 2,540 | 2,507.0 | 2,474 | 2 | 464.8 |
| **C2_Exploitation** | **2,578** | **2,556.5** | **2,535** | 2 | 405.4 |
| C3_Exploration | 2,551 | 2,523.0 | 2,495 | 2 | 350.4 |
| C4_Balanced | 2,461 | 2,449.5 | 2,438 | 2 | 816.0 |
| C5_WidePopulation | 2,557 | 2,510.7 | 2,486 | 3 | 338.3 |

**Best configuration:** C2_Exploitation — highest best score (2,578), highest average (2,556.5), and highest worst-case (2,535)

---

### Instance: `canada_pw` — time-limited (2–3 runs per config)

| Configuration | Best | Avg | Worst | Runs | Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|
| **C1_Baseline** | **2,631** | **2,581.5** | **2,532** | 2 | 330.3 |
| C2_Exploitation | 2,559 | 2,539.5 | 2,520 | 2 | 308.7 |
| C3_Exploration | 2,498 | 2,444.5 | 2,391 | 2 | 372.6 |
| C4_Balanced | 2,464 | 2,459.7 | 2,457 | 3 | 432.4 |
| C5_WidePopulation | 2,551 | 2,519.0 | 2,464 | 3 | 430.0 |

**Best configuration:** C1_Baseline — highest best score (2,631), highest average (2,581.5), and highest worst-case (2,532)

---

### Instance: `usa_tv_input` — time-limited (1–3 runs per config)

| Configuration | Best | Avg | Worst | Runs | Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|
| C1_Baseline | 1,921 | 1,921.0 | 1,921 | 1 | 287.7 |
| C2_Exploitation | 1,711 | 1,711.0 | 1,711 | 2 | 566.8 |
| C3_Exploration | 2,013 | 1,945.0 | 1,877 | 2 | 465.8 |
| **C4_Balanced** | **2,068** | **1,948.0** | 1,828 | 2 | 319.7 |
| C5_WidePopulation | 1,830 | 1,781.7 | 1,716 | 3 | 423.2 |

**Best configuration:** C4_Balanced — highest best score (2,068) and highest average (1,948.0)

---

### Instance: `uk_iptv` — very large (C1–C3 timed out, no results)

> Configurations C1, C2, and C3 each timed out on their first run (single run > 5 minutes). Only C4 and C5 produced valid results.

| Configuration | Best | Avg | Worst | Runs | Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|
| C1_Baseline | — | — | — | 0 | — |
| C2_Exploitation | — | — | — | 0 | — |
| C3_Exploration | — | — | — | 0 | — |
| **C4_Balanced** | **2,529** | **2,484.5** | **2,440** | 2 | 477.3 |
| C5_WidePopulation | 2,502 | 2,502.0 | 2,502 | 1 | 1938.8 |

**Best configuration:** C4_Balanced — the only config to complete multiple runs; C5 completed just 1 run in the 5-minute budget.

---

### Infeasible Instances (exceed 5-minute per-run budget)

The following 4 instances were excluded from experiments — a single run already exceeds the time budget:

| Instance | Approx. channels | Est. time/run |
|---|:---:|:---:|
| `china_pw` | ~1,254 | > 5 min |
| `youtube_gold` | ~1,422 | > 5 min |
| `youtube_premium` | ~1,677 | > 5 min |
| `us_iptv` | ~11,457 | > 45 min |

---

### Overall Ranking (average best-score across 12 comparable instances)

> uk_iptv excluded from ranking: configs C1–C3 produced no results there, making a fair per-config average impossible. Rankings are computed over the 12 instances where all 5 configs have valid data.

| Rank | Configuration | Avg Best Score | Instance wins |
|:---:|---|:---:|:---:|
| 1 | **C4_Balanced** | 1,805.5 | 4 (netherlands, singapore, usa, uk_iptv*) |
| 2 | **C3_Exploration** | 1,799.3 | 2 (kosovo, germany) |
| 3 | **C5_WidePopulation** | 1,797.2 | 2 (croatia, france) |
| 4 | **C1_Baseline** | 1,782.0 | 1 (canada) |
| 5 | **C2_Exploitation** | 1,780.7 | 2 (spain, australia, uk_tv_input*) |

---

## Parameter Analysis

### Effect of α (alpha — pheromone weight)

- **High α (C2: α=3.0):** Ants strongly follow known trails → fast convergence but risk of premature convergence to a local optimum. Observed: C2 achieves the best single score on `spain_iptv` (2,361) and `australia_iptv` (2,578), but has the **highest variance** on `uk_tv_input` (worst=1,247 vs best=1,563), making it unreliable on large instances.
- **Low α (C3: α=1.0):** Pheromone has little influence → decisions driven mostly by heuristic → more diverse solutions across runs. Observed: C3 consistently produces competitive averages on small/medium instances.

### Effect of β (beta — heuristic weight)

- **High β (C3: β=2.0):** Local heuristic quality (program score + bonuses) dominates → good for instances where the heuristic is a strong guide (small/medium instances like germany, kosovo, croatia).
- **Low β (C2: β=0.5):** Heuristic contributes little → pheromone trails and randomness take over → can lead to poor choices when pheromone is not yet well-established.

### Effect of ρ (rho — evaporation rate)

- **High ρ (C3: ρ=0.5):** Trails fade quickly → previous learning is discarded → forces re-exploration each generation. Effective on smaller instances where repeated exploration finds genuine improvements.
- **Low ρ (C2: ρ=0.2):** Trails persist long → strong memory of good solutions → useful for large instances (e.g., `australia_iptv`) where good paths are hard to re-discover.

### Effect of random_factor

- **High (C3: 0.30):** Frequent random jumps introduce strong diversification → guards against local minima → higher run-to-run variance but higher ceiling scores.
- **Low (C2: 0.05):** Ants almost always follow pheromone/heuristic → fast convergence but narrow search.

### Exploration vs. Exploitation Spectrum

```
EXPLORATION <-----------------------------------------> EXPLOITATION
    C3          C4          C1          C5          C2
(a=1,b=2)  (a=b=1.5)  (default)  (ants=150)  (a=3,r=0.2)
```

---

## ACO + Local Search Hybrid Optimization

### Overview

After ACO produces its best solution, a **Hill Climbing Local Search** post-processor is applied once to further improve the result. The hybrid pipeline:

1. Run ACO normally → obtain best ACO solution and score
2. Apply Hill Climbing to the ACO solution → obtain improved solution
3. Compare ACO score vs LS-improved score (delta, percentage improvement)

### Local Search Algorithm: Hill Climbing

Strict hill climbing — only accepting moves that **strictly improve** the current score. Stops when no operator finds an improvement or `max_iterations` is reached (default: 100).

### Neighborhood Operators

**Operator 1 — In-place channel swap**

At each scheduled position `i`, try every other channel whose program starts *and ends* at exactly the same time as the current program. The same-end-time constraint guarantees no cascade effect on the tail of the schedule.

- Also re-evaluates position `i+1` to account for a changed switch penalty.
- Checks all constraints (min_duration, max_consecutive_genre, priority_blocks) before accepting.
- Accepts the swap only if `delta > 0` (strict improvement).

**Operator 2 — Greedy tail rebuild**

Identifies the `max_rebuilds` (default: 5) slots with the lowest fitness scores. For each such slot `i`, removes the entire suffix `scheduled[i:]` and rebuilds it greedily from `scheduled[i].start` using the same greedy heuristic as the ACO fallback. Accepts the rebuild only if the new tail score strictly exceeds the old tail score.

### Stopping Criteria

- No operator finds an improvement in a full iteration → stop immediately
- `max_iterations` outer loop limit reached (default: 100)

### Pipeline Integration

```
ACO (10 ants, 5 generations by default)
    └─> best ACO solution
            └─> Hill Climbing (max 100 iterations)
                    ├─ Operator 1: channel swaps (all positions, each iteration)
                    └─ Operator 2: tail rebuild from worst-fitness slots
```

Each ACO run's LS improvement is tracked independently. The experiment runner records `ls_score`, `ls_improvement`, and `ls_time_sec` per run, and `ls_best`, `ls_avg`, `ls_improvement_avg` per config summary.

### Usage

**Single run with Local Search:**
```bash
python main.py --algo ant --local-search
python main.py --algo ant --local-search --ls-iterations 200
```

**Experiment runner with Local Search (10 runs × 5 configs × all instances):**
```bash
python experiment_runner.py --local-search
python experiment_runner.py --local-search --ls-iterations 150 --instances germany kosovo
```

Output CSV/JSON will include additional LS columns: `ls_score`, `ls_improvement`, `ls_time_sec` (per run) and `ls_best`, `ls_avg`, `ls_improvement_avg`, `ls_total_time_sec` (per config summary).

### Results Summary — ACO vs ACO + Hill Climbing

Best score achieved per instance across all 5 configurations (10 runs each).  
`*` china_pw ACO scores are taken from the ACO+LS experiment run (not included in the standalone ACO experiment).

| Instance | ACO Best | ACO+LS Best | Improvement | Improvement % |
|---|---:|---|---:|---|---:|---:|
| australia_iptv | 2,578 | 2,881 | +303 | +11.8% |
| canada_pw | 2,631 | 3,680 | +1,049 | +39.9% |
| china_pw* | 1,959 | 2,007 | +48 | +2.5% |
| croatia_tv_input | 1,974 | 2,021 | +47 | +2.4% |
| france_iptv | 2,094 | 2,441 | +347 | +16.6% |
| germany_tv_input | 932 | 932 | 0 | 0% |
| kosovo_tv_input | 1,505 | 1,534 | +29 | +1.9% |
| netherlands_tv_input | 1,664 | 1,692 | +28 | +1.7% |
| singapore_pw | 2,642 | 3,318 | +676 | +25.6% |
| spain_iptv | 2,361 | 2,611 | +250 | +10.6% |
| toy | 380 | 380 | 0 | 0% |
| uk_iptv | 2,529 | 3,193 | +664 | +26.3% |
| uk_tv_input | 1,563 | 1,796 | +233 | +14.9% |
| usa_tv_input | 2,068 | 2,571 | +503 | +24.3% |


---
