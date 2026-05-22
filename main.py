import time as _time

from parser.file_selector import select_file
from parser.parser import Parser
from serializer.serializer import SolutionSerializer
from scheduler.beam_search_scheduler import BeamSearchScheduler
from scheduler.ant_scheduler import AntScheduler
from utils.utils import Utils
import argparse


def main():
    parser_arg = argparse.ArgumentParser(description="Run TV scheduling algorithms")
    parser_arg.add_argument("--input", "-i", dest="input_file",
                            help="Path to input JSON (optional)")
    parser_arg.add_argument("--algo", "-a",
                            choices=["beam", "ant"],
                            default="ant",
                            help="Algorithm: beam | ant")
    parser_arg.add_argument("--budget", "-b", type=float, default=8.0,
                            help="Time budget in seconds for SA (default: 8)")
    parser_arg.add_argument("--local-search", action="store_true",
                            help="Apply Hill Climbing Local Search after ACO (ignored for beam)")
    parser_arg.add_argument("--ls-iterations", type=int, default=100,
                            help="Max Hill Climbing iterations (default: 100)")

    args = parser_arg.parse_args()

    file_path = select_file()
    parser    = Parser(file_path)
    instance  = parser.parse()
    Utils.set_current_instance(instance)

    print("\nOpening time:", instance.opening_time)
    print("Closing time:", instance.closing_time)
    n_channels = len(instance.channels)
    print(f"Total Channels: {n_channels}")

    # ------------------------------------------------------------------
    # Algorithm selection
    # ------------------------------------------------------------------

    if args.algo == "beam":
        if n_channels > 50:
            print(f"\n⚠ Beam Search on {n_channels} channels may be very slow.")
        print("\n→ Beam Search Scheduler")
        scheduler = BeamSearchScheduler(
            instance_data=instance,
            beam_width=100,
            lookahead_limit=4,
            density_percentile=25,
            verbose=True,
        )

    else:
        print("\n→ Ant Colony Optimization Scheduler")
        scheduler = AntScheduler(
            instance_data=instance,
            num_ants=10,
            num_generations=5,
            random_seed=42,
            verbose=True,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    t0       = _time.perf_counter()
    solution = scheduler.generate_solution()
    aco_time = _time.perf_counter() - t0
    print(f"\n✓ ACO score: {solution.total_score:,}  (time: {aco_time:.2f}s)")

    # ------------------------------------------------------------------
    # Optional Local Search post-processing
    # ------------------------------------------------------------------

    if args.local_search and args.algo == "ant":
        from scheduler.local_search_scheduler import LocalSearchScheduler
        print(f"\n→ Hill Climbing Local Search (max {args.ls_iterations} iterations)")
        ls = LocalSearchScheduler(
            instance_data  = instance,
            max_iterations = args.ls_iterations,
            verbose        = True,
        )
        t1 = _time.perf_counter()
        solution, stats = ls.improve(solution)
        ls_time = _time.perf_counter() - t1
        improvement = stats["improvement"]
        sign = "+" if improvement >= 0 else ""
        print(f"✓ LS score:  {solution.total_score:,}  "
              f"({sign}{improvement:,}, {sign}{stats['improvement_pct']}%)  "
              f"time: {ls_time:.2f}s")
    elif args.local_search and args.algo != "ant":
        print("\n[!] --local-search is only supported with --algo ant")

    algorithm_name = type(scheduler).__name__.lower()
    if args.local_search and args.algo == "ant":
        algorithm_name += "_ls"
    serializer = SolutionSerializer(
        input_file_path=file_path,
        algorithm_name=algorithm_name,
    )
    serializer.serialize(solution)
    print("✓ Solution saved to output file")


if __name__ == "__main__":
    main()