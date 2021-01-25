from .colors import COLORS
from tqdm.auto import tqdm

from collections import defaultdict

from jammy.utils.meter import GroupMeters


class CmdLineViz:
    def __init__(self):
        self.meter = defaultdict(GroupMeters)
        self.prev_mean = {}

    def update(self, mode, eval_dict):
        self.meter[mode].update(eval_dict)

    def flush(self):
        if len(self.meter) == 0:
            return

        _str = "\n" + "\t" * 1 + "=== Summary ===\n"
        keys = []
        color_write = {}
        for mode, meters in self.meter.items():
            avg = meters.avg
            color_write[mode] = defaultdict(lambda: " " * 10)
            keys.extend(list(avg.keys()))
            for key, value in avg.items():
                color = COLORS.White
                if mode + key in self.prev_mean:
                    color = (
                        COLORS.Green
                        if value > self.prev_mean[mode + key]
                        else COLORS.Red
                    )
                color_write[mode][key] = f"{color}{value:10.4f}{COLORS.END_NO_TOKEN}"
                self.prev_mean[mode + key] = value

        keys = set(keys)
        _str += f"\t{'mode':<10}"
        for cur_key in self.meter:
            _str += f" -- {cur_key:>10}"
        _str += "\n"
        for cur_key in keys:
            _str += f"\t{cur_key:10}"
            for _, writer in color_write.items():
                _str += f" -- {writer[cur_key]}"
            _str += f"\n"
        tqdm.write(_str + "\n")

        self.meter = defaultdict(GroupMeters)


if __name__ == "__main__":
    import numpy as np

    N = 10
    line = CmdLineViz()
    increase = np.arange(N) + np.random.randn(N)
    decrease = -np.arange(N) + np.random.randn(N)

    for cur_inc, cur_dec in zip(increase, decrease):
        line.update("test", {"increase": cur_inc, "decrease": cur_dec})

    line.flush()

    for cur_inc, cur_dec in zip(decrease, increase):
        line.update("test", {"increase": cur_inc, "decrease": cur_dec})

    line.flush()
