from parser.file_selector import select_file
from parser.parser import Parser
from serializer.serializer import SolutionSerializer
from scheduler.beam_search_scheduler import BeamSearchScheduler
from scheduler.genetic_scheduler import GeneticScheduler
from scheduler.simulated_annealing_scheduler import SimulatedAnnealingScheduler
from utils.utils import Utils
import argparse


def main():
    parser_arg = argparse.ArgumentParser(description="Run TV scheduling algorithms")
    parser_arg.add_argument("--input", "-i", dest="input_file",
                            help="Path to input JSON (optional)")
    parser_arg.add_argument("--algo", "-a",
                            choices=["beam", "sa", "ga"],
                            default="sa",
                            help="Algorithm: beam | sa (default) | ga")
    parser_arg.add_argument("--budget", "-b", type=float, default=8.0,
                            help="Time budget in seconds for SA (default: 8)")

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

    elif args.algo == "ga":
        print("\n→ Genetic Algorithm Scheduler")
        pop_size = max(20, min(50, 6000 // n_channels))
        max_gen  = max(60, min(400, 20000 // n_channels))
        scheduler = GeneticScheduler(
            instance_data=instance,
            pop_size=pop_size,
            max_generations=max_gen,
            elite_size=5,
            crossover_rate=0.80,
            mutation_rate=0.40,
            patience=60,
            random_seed=42,
            verbose=True,
        )

    else:  # sa  (default — fast, best score for budget)
        print(f"\n→ Time-Boxed Hybrid Scheduler  (budget={args.budget}s)")
        scheduler = SimulatedAnnealingScheduler(
            instance_data=instance,
            time_budget=args.budget,
            pop_size=40,
            random_seed=42,
            verbose=True,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    solution = scheduler.generate_solution()
    print(f"\n✓ Generated solution with total score: {solution.total_score}")

    algorithm_name = type(scheduler).__name__.lower()
    serializer = SolutionSerializer(
        input_file_path=file_path,
        algorithm_name=algorithm_name,
    )
    serializer.serialize(solution)
    print("✓ Solution saved to output file")


if __name__ == "__main__":
    main()