from math import sqrt

import matplotlib as mpl
import matplotlib.pyplot as plt

COLORS = [
    "#3F51B5",  # Indigo
    "#00BCD4",  # Cyan
    "#FF1493",  # Light Pink
    "#9C27B0",  # Purple
    "#FF5722",  # Deep Orange
    "#FF9800",  # Orange
    "#4CAF50",  # Strong Green
    "#2196F3",  # Blue
    "#FFC107",  # Amber
    "#E91E63",  # Pink
    "#8BC34A",  # Light Green
]


mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLORS)
mpl.rcParams.update({"font.size": 20})
mpl.rcParams.update({"figure.titlesize": 30})
mpl.rcParams.update({"axes.titlesize": 25})
mpl.rcParams.update({"axes.labelsize": 20})
mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["figure.autolayout"] = True
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.linestyle"] = "--"
mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["grid.color"] = "k"
mpl.rcParams["font.family"] = "serif"
scale = 6
mpl.rcParams["figure.figsize"] = [scale * (1 + sqrt(5)) / 2, scale]
mpl.rcParams["figure.facecolor"] = "w"
plt.rcParams.update(mpl.rcParams)


def color_palette():
    """
    Display a beautiful color palette to choose from.
    """
    _, ax = plt.subplots(1, len(COLORS), figsize=(20, 2.5))
    for i, color in enumerate(COLORS):
        ax[i].set_facecolor(color)
        ax[i].set_title(i)
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0, 1)
        ax[i].xaxis.set_visible(False)
        ax[i].yaxis.set_visible(False)

    plt.tight_layout()
    plt.show()
