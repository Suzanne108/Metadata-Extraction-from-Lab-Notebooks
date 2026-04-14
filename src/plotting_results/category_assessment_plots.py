from .plotting_results import plot_category_assessment
from .plotting_results_stacked import plot_category_assessment_stacked


def get_category_assessment_datasets() -> list[dict]:
    """Return the datasets used for the category-assessment plots.

    This keeps data definition separate from plotting so that
    multiple visualisations (grouped, stacked, etc.) can reuse
    the same underlying values.
    """

    return [
        {
            "label": "Aggregate",
            "good":        [15, 28, 16, 8, 44, 9, 13, 5],
            "sufficient":  [10, 15, 2, 21, 19, 18, 3, 2],
            "insufficient":[5, 2, 0, 13, 1, 17, 4, 0],
            "missing":     [0, 13, 2, 3, 23, 22, 4, 0],
            "save_path": "category_assessment_aggregate.png",
        },
        {
            "label": "N1",
            "good":        [3, 2, 1, 4, 5, 6, 7, 1],
            "sufficient":  [1, 3, 0, 0, 1, 0, 0, 0],
            "insufficient":[0, 1, 0, 0, 1, 0, 0, 0],
            "missing":     [0, 6, 0, 0, 0, 5, 0, 0],
            "save_path": "category_assessment_experiment_1.png",
        },
        {
            "label": "N2",
            "good":        [2, 5, 1, 0, 9, 2, 2, 1],
            "sufficient":  [1, 2, 0, 5, 0, 4, 1, 0],
            "insufficient":[1, 0, 0, 0, 1, 1, 1, 0],
            "missing":     [0, 2, 0, 0, 2, 2, 3, 0],
            "save_path": "category_assessment_experiment_2.png",
        },
        {
            "label": "N3",
            "good":        [0, 3, 0, 2, 8, 1, 1, 0],
            "sufficient":  [3, 4, 0, 3, 3, 3, 0, 1],
            "insufficient":[1, 0, 0, 5, 0, 1, 1, 0],
            "missing":     [0, 2, 2, 0, 6, 4, 1, 0],
            "save_path": "category_assessment_experiment_3.png",
        },
        {
            "label": "N4",
            "good":        [3, 6, 11, 0, 7, 0, 1, 1],
            "sufficient":  [3, 2, 1, 0, 9, 5, 2, 0],
            "insufficient":[0, 0, 0, 8, 0, 2, 0, 0],
            "missing":     [0, 0, 0, 0, 8, 1, 0, 0],
            "save_path": "category_assessment_experiment_4.png",
        },
        {
            "label": "N5",
            "good":        [3, 4, 1, 2, 6, 0, 2, 1],
            "sufficient":  [0, 2, 0, 0, 1, 0, 0, 0],
            "insufficient":[1, 0, 0, 3, 0, 0, 0, 0],
            "missing":     [0, 2, 0, 0, 1, 8, 0, 0],
            "save_path": "category_assessment_experiment_5.png",
        },
        {
            "label": "N6",
            "good":        [3, 6, 1, 0, 5, 0, 0, 0],
            "sufficient":  [0, 0, 0, 0, 4, 3, 0, 1],
            "insufficient":[1, 1, 0, 0, 0, 3, 1, 0],
            "missing":     [0, 0, 0, 0, 2, 0, 0, 0],
            "save_path": "category_assessment_experiment_6.png",
        },
        {
            "label": "N7",
            "good":        [1, 2, 1, 0, 4, 0, 0, 1],
            "sufficient":  [2, 2, 1, 13, 1, 3, 0, 0],
            "insufficient":[1, 0, 0, 0, 0, 10, 1, 0],
            "missing":     [0, 1, 0, 0, 3, 2, 0, 0],
            "save_path": "category_assessment_experiment_7.png",
        },
    ]


def plot_results_grouped() -> None:
    """Plot the category-assessment results with side-by-side bars."""

    for dataset in get_category_assessment_datasets():
        plot_category_assessment(
            dataset["good"],
            dataset["sufficient"],
            dataset["insufficient"],
            dataset["missing"],
            title=f"Assessment of extracted metadata - {dataset['label']}",
            save_path=dataset["save_path"],
        )


def plot_results_stacked() -> None:
    """Plot the category-assessment results with stacked bars."""

    for dataset in get_category_assessment_datasets():
        # Reuse the same base filename but clarify stacked output
        base_path = dataset["save_path"]
        if base_path.lower().endswith(".png"):
            stacked_path = base_path[:-4] + "_stacked.png"
        else:
            stacked_path = base_path + "_stacked.png"

        plot_category_assessment_stacked(
            dataset["good"],
            dataset["sufficient"],
            dataset["insufficient"],
            dataset["missing"],
            title=f"Assessment of extracted metadata - {dataset['label']}",
            save_path=stacked_path,
        )
