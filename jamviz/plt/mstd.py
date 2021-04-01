import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from collections import defaultdict
from .color_list import jcolors

__all__ = ["MstdDict", "mstd_plot", "plotstd"]

color_list = jcolors


def MstdDict():
    return defaultdict(lambda: defaultdict(list))


def point_stat(data, coef_std=1, **kwargs):
    mean = np.mean(data)
    if "ylog" in kwargs:
        std = np.std(data)
        up = np.log(mean + std)
        down = np.log(mean - std)
        space = coef_std * np.min([up - np.log(mean), np.log(mean) - down])
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


def plotstd(mean, cov, x=None, color=None, ax=None, label=None, markersize=3):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if color is None:
        color = color_list[0]

    mean = np.array(mean).flatten()
    cov = np.array(cov).flatten()
    if x is None:
        x = np.arange(len(mean))

    ax.plot(x, mean, "o", color=color, markersize=markersize)
    ax.plot(x, mean, "-", color=color, label=label)
    ax.fill_between(x, mean - cov, mean + cov, color=color, alpha=0.2)

    return ax


def mstd_plot(
    exp_data, labels=None, size=(7, 7), ax=None, is_label=True, coef_std=1, **kwargs
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
    """
    global color_list
    if labels == None:
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
        up = rtn_dict[exp_name]["up"]
        down = rtn_dict[exp_name]["down"]
        ax.plot(x, mean, "o", color=color_list[i], markersize=12)
        ax.plot(x, mean, "-", color=color_list[i])
        ax.fill_between(x, down, up, color=color_list[i], alpha=0.2)

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
