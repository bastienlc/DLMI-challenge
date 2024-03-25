import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_validate


def custom_cross_validate(model: BaseEstimator, X: np.ndarray, y: np.ndarray, k_fold=5):
    def scoring_function(model, X, y):
        y_pred = model.predict(X)
        return balanced_accuracy_score(y, y_pred)

    scores = cross_validate(
        model, X, y, cv=k_fold, scoring=scoring_function, return_train_score=True
    )

    return scores["train_score"], scores["test_score"]


def plot_comparison(models_scores, labels):
    _, ax = plt.subplots(figsize=(20, 10))
    ax.boxplot(models_scores, labels=labels, meanline=True, showmeans=True)
    ax.hlines(
        0.85714, 0, len(models_scores) + 1, linestyles="dashed", label="Reference score"
    )
    ax.set_yticks(np.arange(0, 1.1, 0.05))
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.legend()
    plt.show()


def compare_models(models, X, y, k_fold=5, plot=True, verbose=False):
    labels = []
    models_scores = []
    for k, (model, model_name) in enumerate(models):
        print(
            f"Evaluating model {k+1}/{len(models)} ({model_name})",
            end="\r",
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
