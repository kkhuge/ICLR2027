"""Shared publication style for all paper figures."""

import matplotlib as mpl


PLOT_FONT_SIZE = 16


def apply_plot_style():
    """Apply a consistent Times-like serif style, including math text."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": PLOT_FONT_SIZE,
            "axes.titlesize": PLOT_FONT_SIZE,
            "axes.labelsize": PLOT_FONT_SIZE,
            "figure.titlesize": PLOT_FONT_SIZE,
            "legend.fontsize": PLOT_FONT_SIZE,
            "xtick.labelsize": PLOT_FONT_SIZE,
            "ytick.labelsize": PLOT_FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
