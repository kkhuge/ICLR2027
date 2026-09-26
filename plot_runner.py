"""Run a plotting script and save its displayed figures as tightly cropped PDFs."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from plot_style import apply_plot_style, PLOT_FONT_SIZE


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
        default=None,
        help="Override the paper's crop with an automatic tight crop and this margin",
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
            fig = plt.figure(number)
            apply_plot_style()
            fig.set_size_inches(5, 4)
            fig.canvas.draw()
            for text in fig.findobj(match=Text):
                text.set_fontfamily("STIXGeneral")
                text.set_math_fontfamily("stix")
                text.set_fontsize(PLOT_FONT_SIZE)
            fig.tight_layout()
            # Match the crops used by the current paper. Figure 1 has a
            # separately updated crop; the other panels share one crop.
            points = ((15, 18, 345, 273) if script.stem in (
                "loss_femnist_niid_client_level", "acc_femnist_niid_client_level"
            ) else (15.28, 15.28, 344.72, 272.72))
            crop = Bbox.from_extents(*(value / 72 for value in points))
            fig.savefig(
                target,
                bbox_inches=crop if args.pad_inches is None else "tight",
                pad_inches=0 if args.pad_inches is None else args.pad_inches,
            )
            print(f"Saved {target}", flush=True)
        if args.show:
            return original_show(*show_args, **show_kwargs)
        return None

    plt.show = save_then_show
    source = script.read_text(encoding="utf-8-sig")
    if not args.show:
        source = source.replace('matplotlib.use("TkAgg")', 'matplotlib.use("Agg")')
    exec(compile(source, str(script), "exec"),
         {"__name__": "__main__", "__file__": str(script)})


if __name__ == "__main__":
    main()
