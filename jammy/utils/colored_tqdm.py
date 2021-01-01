from tqdm.auto import tqdm
import collections
from tqdm import std
import numpy as np

from jammy.cli.colors import COLORS

from jammy.utils.meter import GroupMeters


class Coloredtqdm(tqdm):
    def __init__(self, *kargs, **kwargs):
        super(Coloredtqdm, self).__init__(*kargs, **kwargs)
        self._meter = GroupMeters()

    def set_postfix(
        self, ordered_dict=None, refresh=True, color=True, round=4, **kwargs
    ):
        postfix = std._OrderedDict([] if ordered_dict is None else ordered_dict)
        value_color = collections.defaultdict(lambda: COLORS.White)
        prev_read = self._meter.avg

        for key in sorted(kwargs.keys()):
            postfix[key] = kwargs[key]

        for key in postfix.keys():
            if key in prev_read:
                if prev_read[key] > postfix[key]:
                    value_color[key] = COLORS.Red
                else:
                    value_color[key] = COLORS.Green

        self._meter.update(postfix)

        for key in postfix.keys():
            if isinstance(postfix[key], std.Number):
                postfix[key] = self.format_num_to_k(
                    np.round(postfix[key], round), k=round + 1
                )
            if isinstance(postfix[key], std._basestring):
                postfix[key] = str(postfix[key])
            if len(postfix[key]) != round:
                postfix[key] += (round - len(postfix[key])) * " "

        if color:
            self.postfix = ", ".join(
                value_color[key] + key + "=" + postfix[key] + COLORS.END_NO_TOKEN
                for key in postfix.keys()
            )
        else:
            self.postfix = ", ".join(key + "=" + postfix[key] for key in postfix.keys())

        if refresh:
            self.refresh()

    def format_num_to_k(self, seq, k=4):
        seq = str(seq)
        length = len(seq)
        out = seq + " " * (k - length) if length < k else seq
        return out if length < k else seq[:k]
