"""
Generate one output JSON per feasible instance using the best-performing
ACO configuration from experiments.
"""
import sys
import io
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from parser.parser import Parser
from scheduler.ant_scheduler import AntScheduler
from serializer.serializer import SolutionSerializer
from utils.utils import Utils

# Best config per instance (from experiments)
INSTANCE_CONFIGS = {
    "toy":                 dict(num_ants=100, num_generations=10, alpha=2.0, beta=1.0, rho=0.30, random_factor=0.15),  # C1 (all tied)
    "germany_tv_input":    dict(num_ants=80,  num_generations=15, alpha=1.0, beta=2.0, rho=0.50, random_factor=0.30),  # C3
    "netherlands_tv_input":dict(num_ants=60,  num_generations=15, alpha=1.5, beta=1.5, rho=0.35, random_factor=0.10),  # C4
    "kosovo_tv_input":     dict(num_ants=80,  num_generations=15, alpha=1.0, beta=2.0, rho=0.50, random_factor=0.30),  # C3
    "croatia_tv_input":    dict(num_ants=150, num_generations=5,  alpha=2.0, beta=1.5, rho=0.25, random_factor=0.20),  # C5
    "spain_iptv":          dict(num_ants=50,  num_generations=20, alpha=3.0, beta=0.5, rho=0.20, random_factor=0.05),  # C2
    "france_iptv":         dict(num_ants=150, num_generations=5,  alpha=2.0, beta=1.5, rho=0.25, random_factor=0.20),  # C5
    "singapore_pw":        dict(num_ants=60,  num_generations=15, alpha=1.5, beta=1.5, rho=0.35, random_factor=0.10),  # C4
    "uk_tv_input":         dict(num_ants=50,  num_generations=20, alpha=3.0, beta=0.5, rho=0.20, random_factor=0.05),  # C2
    "australia_iptv":      dict(num_ants=50,  num_generations=20, alpha=3.0, beta=0.5, rho=0.20, random_factor=0.05),  # C2
    "canada_pw":           dict(num_ants=100, num_generations=10, alpha=2.0, beta=1.0, rho=0.30, random_factor=0.15),  # C1
    "usa_tv_input":        dict(num_ants=60,  num_generations=15, alpha=1.5, beta=1.5, rho=0.35, random_factor=0.10),  # C4
    "uk_iptv":             dict(num_ants=60,  num_generations=15, alpha=1.5, beta=1.5, rho=0.35, random_factor=0.10),  # C4
}

INPUT_DIR = Path("data/input")

def run_instance(name: str, cfg: dict) -> None:
    path = INPUT_DIR / f"{name}.json"
    if not path.exists():
        print(f"  [SKIP] {name}.json not found")
        return

    print(f"\n  Running {name} ...", flush=True)
    instance_data = Parser(str(path)).parse()
    Utils.set_current_instance(instance_data)

    scheduler = AntScheduler(
        instance_data=instance_data,
        random_seed=42,
        verbose=False,
        **cfg,
    )
    solution = scheduler.generate_solution()
    print(f"  Score: {solution.total_score:,}", flush=True)

    serializer = SolutionSerializer(
        input_file_path=str(path),
        algorithm_name="antscheduler",
    )
    serializer.serialize(solution)


if __name__ == "__main__":
    print("Generating output files for all feasible instances...")
    for instance_name, config in INSTANCE_CONFIGS.items():
        run_instance(instance_name, config)
    print("\nDone.")
