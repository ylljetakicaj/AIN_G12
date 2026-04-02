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