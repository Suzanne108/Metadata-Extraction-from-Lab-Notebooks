import matplotlib.pyplot as plt
import numpy as np

def plot_category_assessment(
    good,
    sufficient,
    insufficient,
    missing,
    categories=None,
    title="Assessment of extracted metadata by ISA object type",
    save_path=None,
    dpi=300
):
    """
    Create a publication-quality grouped bar chart.

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
            "Factors"
        ]

    data_length = len(categories)

    for dataset in [good, sufficient, insufficient, missing]:
        if len(dataset) != data_length:
            raise ValueError("All data lists must match the number of categories.")

    x = np.arange(data_length)
    width = 0.18

    # Publication-friendly style
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {
        "Good": "#2E7D32",        # muted green
        "Sufficient": "#F9A825",  # rich amber
        "Insufficient": "#C62828",# deep red
        "Missing": "#CFD8DC",     # light blue-grey for readability
    }

    bars_good = ax.bar(x - 1.5 * width, good, width, label="Good", color=colors["Good"])
    bars_sufficient = ax.bar(x - 0.5 * width, sufficient, width, label="Sufficient", color=colors["Sufficient"])
    bars_insufficient = ax.bar(x + 0.5 * width, insufficient, width, label="Insufficient", color=colors["Insufficient"])
    bars_missing = ax.bar(x + 1.5 * width, missing, width, label="Missing", color=colors["Missing"])

    ax.set_xlabel("ISA object type", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")

    ax.legend(frameon=False)

    # Remove top/right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9
            )

    for bar_group in [bars_good, bars_sufficient, bars_insufficient, bars_missing]:
        add_labels(bar_group)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()