"""Run a plotting script and save its displayed figures as tightly cropped PDFs."""

import argparse
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure_output"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", help="One plotting script in this directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT,
        help="Directory for generated PDFs (default: figure_output)",
    )
    parser.add_argument(
        "--pad-inches",
        type=float,
        default=0.02,
        help="Safety margin around the tight bounding box (default: 0.02)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures after saving (disabled by default for batch runs)",
    )
    args = parser.parse_args()
    script = (ROOT / args.script).resolve()
    if script.parent != ROOT or script.suffix != ".py" or not script.exists():
        raise ValueError("The plot script must be a .py file in the release directory")

    output_dir = args.output_dir.resolve()
    original_show = plt.show

    def save_then_show(*show_args, **show_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_numbers = plt.get_fignums()
        for index, number in enumerate(figure_numbers, start=1):
            suffix = "" if len(figure_numbers) == 1 else f"_{index}"
            target = output_dir / f"{script.stem}{suffix}.pdf"
            plt.figure(number).savefig(
                target,
                bbox_inches="tight",
                pad_inches=args.pad_inches,
            )
            print(f"Saved {target}", flush=True)
        if args.show:
            return original_show(*show_args, **show_kwargs)
        return None

    plt.show = save_then_show
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
