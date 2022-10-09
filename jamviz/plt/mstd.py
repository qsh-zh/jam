from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .color_list import jcolors

__all__ = ["MstdDict", "mstd_plot", "plotstd"]

color_list = jcolors


def MstdDict():
    return defaultdict(lambda: defaultdict(list))


# pylint: disable=too-many-locals
def point_stat(data, coef_std=1, **kwargs):
    mean = np.mean(data)
    if "ylog" in kwargs:
        std = np.std(data)
        above = np.log(mean + std)
        down = np.log(mean - std)
        space = coef_std * np.min([above - np.log(mean), np.log(mean) - down])
        return {
            "mean": mean,
            "up": np.exp(np.log(mean) + space),
            "down": np.exp(np.log(mean) - space),
        }
    else:
        return {
            "mean": mean,
            "up": mean + coef_std * np.std(data),
            "down": mean - coef_std * np.std(data),
        }


def plotstd(mean, cov, x=None, color=None, ax=None, label=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if color is None:
        color = color_list[0]

    mean = np.array(mean).flatten()
    cov = np.array(cov).flatten()
    if x is None:
        x = np.arange(len(mean))

    ax.plot(x, mean, "o", color=color)
    ax.plot(x, mean, "-", color=color, label=label)
    ax.fill_between(x, mean - cov, mean + cov, color=color, alpha=0.2)

    return ax


def mstd_plot(  # pylint: disable=too-many-branches
    exp_data,
    labels=None,
    size=(7, 7),
    ax=None,
    is_label=True,
    coef_std=1,
    marker=None,
    **kwargs,
):
    """plot mean and std of sequence

    :param exp_data: loaded data, key:method->key:x_value->list of y
    :type exp_data: dict
    :param labels: key:method->value:figure labels, or None
    :type labels: dict
    :param size: figure_size, defaults to (7, 7)
    :param ax: maptlotlib axis, defaults to None
    :param is_label: whether or not show label, defaults to True
    :param coef_std: width of std, defaults to 1
    :return: ax

    Example::
        exp_data = {
            "method_A": {
                            1: [seed1_y, seed2_y, seed3_y],
                            2: [seed1_y, seed2_y, seed3_y],
                        },
            "method_B": {
                            1: [seed1_y, seed2_y, seed3_y],
                            2: [seed1_y, seed2_y, seed3_y]
                        }
        }
        labels = {
            "method_A": "name_A_in_paper",
            "method_B": "name_B_in_paper",
        }
    """
    global color_list  # pylint: disable=global-variable-not-assigned
    if marker is None:
        marker = [
            "-",
        ]
    if labels is None:
        labels = {key_: key_ for key_ in exp_data.keys()}
    rtn_dict = {}
    for exp_name in labels.keys():
        cur = []
        for _y_list in exp_data[exp_name].values():
            cur.append(list(point_stat(_y_list, coef_std, **kwargs).values()))

        cur = np.array(cur)
        rtn_dict[exp_name] = {
            "x": np.array(list(exp_data[exp_name].keys())),
            "mean": cur[:, 0],
            "up": cur[:, 1],
            "down": cur[:, 2],
        }

    if ax is None:
        _, ax = plt.subplots(figsize=size)
    for i, exp_name in enumerate(labels):
        x = rtn_dict[exp_name]["x"]
        mean = rtn_dict[exp_name]["mean"]
        above = rtn_dict[exp_name]["up"]
        down = rtn_dict[exp_name]["down"]
        cur_color = color_list[i % len(color_list)]
        # ax.plot(x, mean, "o", color=cur_color)
        # ax.plot(x, mean, "-", color=cur_color)
        for cur_marker in marker:
            ax.plot(x, mean, cur_marker, color=cur_color)
        ax.fill_between(x, down, above, color=color_list[i], alpha=0.2)

    if "fix_legend" in kwargs:
        legend_label = [f"{kwargs['fix_legend']}={value}" for value in labels.values()]
    else:
        legend_label = list(labels.values())

    if is_label:
        custom_lines = [
            Line2D([0], [0], color=color_list[i], lw=4) for i, _ in enumerate(labels)
        ]
        #        ax.legend(custom_lines, list(labels.values()))
        ax.legend(custom_lines, legend_label)

    if "xlabel" in kwargs:
        ax.set_xlabel(kwargs["xlabel"])
    if "ylabel" in kwargs:
        ax.set_ylabel(kwargs["ylabel"])
    if "ylog" in kwargs:
        ax.set_yscale("log")

    fig = plt.gcf()

    return fig, ax
