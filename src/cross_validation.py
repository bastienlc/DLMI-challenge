import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from style import COLORS

from .config import CONFIG


def custom_cross_validate(model: BaseEstimator, X: np.ndarray, y: np.ndarray, k_fold=5):
    def scoring_function(model, X, y):
        y_pred = model.predict(X)
        return balanced_accuracy_score(y, y_pred)

    folder = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=CONFIG.SEED)

    for i, (train_index, test_index) in enumerate(folder.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        train_score = scoring_function(model, X_train, y_train)
        test_score = scoring_function(model, X_test, y_test)

        if i == 0:
            scores = {
                "train_score": [train_score],
                "test_score": [test_score],
            }
        else:
            scores["train_score"].append(train_score)
            scores["test_score"].append(test_score)

    return scores["train_score"], scores["test_score"]


def plot_comparison(models_scores, labels):
    _, ax = plt.subplots(figsize=(40, 20))
    box_plot = ax.boxplot(
        models_scores,
        labels=labels,
        meanline=True,
        showmeans=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightpink"),
    )
    for median in box_plot["medians"]:
        median.set_color(COLORS[3])
        median.set_linewidth(2)
    for mean in box_plot["means"]:
        mean.set_color(COLORS[6])
        mean.set_linewidth(2)
        mean.set_linestyle("solid")
    ax.hlines(
        0.85714,
        0,
        len(models_scores) + 1,
        linestyles="dashed",
        label="Baseline test score",
        color=COLORS[9],
    )
    ax.plot([], [], color=COLORS[3], label="Median")
    ax.plot([], [], color=COLORS[6], label="Mean")
    ax.set_yticks(np.arange(0, 1.1, 0.05))
    ax.set_ylim(0.5, 1.0)
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.legend()
    plt.tight_layout()
    plt.savefig("report/figures/unsupervised_classification_models.png")
    plt.show()


def compare_models(models, X, y, k_fold=5, plot=True, verbose=False):
    labels = []
    models_scores = []
    for k, (model, model_name) in enumerate(models):
        print(
            f"Evaluating model {k+1}/{len(models)} ({model_name})",
        )
        try:
            _, validation_scores = custom_cross_validate(model, X, y, k_fold=k_fold)
            labels.append(model_name)
            models_scores.append(validation_scores)
        except Exception as e:
            print()
            print("Failed on model ", model_name)
            if verbose:
                print(e)

    print()

    models_scores = np.nan_to_num(models_scores)
    min_scores = np.min(models_scores, axis=1)
    sorter = np.argsort(min_scores)[::-1]
    models_scores, labels = (
        [models_scores[i] for i in sorter],
        [labels[i] for i in sorter],
    )

    if plot:
        plot_comparison(models_scores, labels)

    return models_scores, labels
