import matplotlib.pyplot as plt
import numpy as np


def plot_category_assessment_stacked(
    good,
    sufficient,
    insufficient,
    missing,
    categories=None,
    title="Assessment of extracted metadata by ISA object type (stacked)",
    save_path=None,
    dpi=300,
):
    """Create a publication-quality stacked bar chart.

    Parameters
    ----------
    good : list
        Counts for 'Good'
    sufficient : list
        Counts for 'Sufficient'
    insufficient : list
        Counts for 'Insufficient'
    missing : list
        Counts for 'Missing'
    categories : list
        Category labels for the x-axis
    title : str
        Plot title
    save_path : str
        File path to save the figure (optional)
    dpi : int
        Resolution for saved figure
    """

    if categories is None:
        categories = [
            "Title/Description",
            "Protocols",
            "Sources",
            "Samples",
            "Materials",
            "Processes",
            "Assays",
            "Factors",
        ]

    data_length = len(categories)

    for dataset in [good, sufficient, insufficient, missing]:
        if len(dataset) != data_length:
            raise ValueError("All data lists must match the number of categories.")

    x = np.arange(data_length)

    # Publication-friendly style
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {
        "Good": "#2E7D32",        # muted green
        "Sufficient": "#F9A825",  # rich amber
        "Insufficient": "#C62828",# deep red
        "Missing": "#CFD8DC",     # light blue-grey for readability
    }

    # Convert raw counts to percentages per category so that each
    # stacked bar sums to 100%.
    good = np.array(good, dtype=float)
    sufficient = np.array(sufficient, dtype=float)
    insufficient = np.array(insufficient, dtype=float)
    missing = np.array(missing, dtype=float)

    totals = good + sufficient + insufficient + missing
    # Avoid division by zero: where total is 0, keep all percentages at 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        good_pct = np.where(totals > 0, (good / totals) * 100, 0)
        sufficient_pct = np.where(totals > 0, (sufficient / totals) * 100, 0)
        insufficient_pct = np.where(totals > 0, (insufficient / totals) * 100, 0)
        missing_pct = np.where(totals > 0, (missing / totals) * 100, 0)

    bars_good = ax.bar(x, good_pct, label="Good", color=colors["Good"])
    bars_sufficient = ax.bar(
        x,
        sufficient_pct,
        bottom=good_pct,
        label="Sufficient",
        color=colors["Sufficient"],
    )
    bars_insufficient = ax.bar(
        x,
        insufficient_pct,
        bottom=good_pct + sufficient_pct,
        label="Insufficient",
        color=colors["Insufficient"],
    )
    bars_missing = ax.bar(
        x,
        missing_pct,
        bottom=good_pct + sufficient_pct + insufficient_pct,
        label="Missing",
        color=colors["Missing"],
    )

    ax.set_xlabel("ISA object type", fontsize=12)
    ax.set_ylabel("Percentage of objects (%)", fontsize=12)
    ax.set_title(title, fontsize=14, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")

    # Place legend slightly to the right and near the bottom,
    # outside the main plotting area for readability.
    ax.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.02),
        borderaxespad=0.0,
    )

    # Remove top/right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    def add_labels_on_segments(bars, raw_values):
        """Annotate each bar segment with its raw count."""

        for bar, raw in zip(bars, raw_values):
            height = bar.get_height()
            if height == 0:
                continue
            ax.annotate(
                f"{int(raw)}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=9,
            )

    add_labels_on_segments(bars_good, good)
    add_labels_on_segments(bars_sufficient, sufficient)
    add_labels_on_segments(bars_insufficient, insufficient)
    add_labels_on_segments(bars_missing, missing)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()