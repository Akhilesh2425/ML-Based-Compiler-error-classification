
import os
import sys
import subprocess
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = {
    1: ("preprocess.py",            "Clean dataset & balance classes"),
    2: ("train_supervisied.py",     "Train supervised models (LR / NB / SVM)"),
    3: ("train_semi_supervised.py", "Self-training on unlabeled data"),
    4: ("cli_demo.py",              "Interactive CLI demo"),
}


def run_step(step_num: int):
    filename, desc = STEPS[step_num]
    script = os.path.join(BASE_DIR, filename)

    print(f"\n{'='*55}")
    print(f"  Step {step_num}: {desc}")
    print(f"{'='*55}")

    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"\n Step {step_num} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n Step {step_num} complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Compiler Error Classification Pipeline"
    )
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3, 4],
        help="Run only a specific step (1-4)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Launch CLI demo directly (Step 4)"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  Compiler Error Classification System")
    print("  Roll: 24CSB0A25")
    print("=" * 55)

    print("\nPipeline Steps:")
    for num, (_, desc) in STEPS.items():
        print(f"  Step {num}: {desc}")

    if args.demo:
        run_step(4)
        return

    if args.step:
        run_step(args.step)
        return

    # Run full pipeline steps 1–3 (step 4 is interactive)
    for step_num in [1, 2, 3]:
        run_step(step_num)

    print(f"\n{'='*55}")
    print("  Pipeline complete!")
    print("  Run:  python main.py --demo")
    print("   or:  python cli_demo.py")
    print("  to launch the interactive classifier.")
    print("="*55)


if __name__ == "__main__":
    main()