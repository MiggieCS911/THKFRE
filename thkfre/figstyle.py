"""Publication figure style for BMC Nephrology (Springer Nature / BMC rules).

BMC: width 85 mm (single column) or 170 mm (double), max height 225 mm,
~300 dpi at final size, lines > 0.25 pt, fonts embedded, PDF preferred.
Figure titles and legends go in the manuscript, keys go inside the figure.

Colours: validated categorical palette (colour-vision-deficiency safe for these
three slots, checked with the dataviz validator). Each model ALSO has its own line
style and marker, so figures stay readable in greyscale print.
"""
import matplotlib as mpl

MM = 1 / 25.4                       # inches per mm
SINGLE = 85 * MM
DOUBLE = 170 * MM
DPI = 300

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"

# entity -> fixed style (same in every figure)
MODEL_STYLE = {
    "TH-KFRE v1.0": {"color": "#2a78d6", "ls": "-", "marker": "o", "label": "TH-KFRE"},
    "KFRE-nonNA": {"color": "#eb6834", "ls": "--", "marker": "s", "label": "KFRE (non-North American)"},
    "KFRE-NA": {"color": "#1baf7a", "ls": "-.", "marker": "^", "label": "KFRE (North American)"},
}
OUTCOME_STYLE = {
    "KF": {"color": "#2a78d6", "ls": "-", "label": "Kidney failure"},
    "Death": {"color": INK_2, "ls": "--", "label": "Death before kidney failure"},
}


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "legend.frameon": False, "lines.linewidth": 1.2,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType fonts
        "savefig.dpi": DPI,
    })


def panel_label(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def save(fig, path_stem):
    """Save as vector PDF (preferred by BMC) plus 300-dpi TIFF (LZW) and PNG preview."""
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{path_stem}.tiff", dpi=DPI, bbox_inches="tight", pad_inches=0.02,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(f"{path_stem}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.02)
