from .colors import COLORS
from tqdm import tqdm

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
        
        tqdm.write("\t"*3 + "=== Summary ===")
        for mode, meters in self.meter.items():
            avg = meters.avg
            for key,value in avg.items():
                color = COLORS.White
                if mode+key in self.prev_mean:
                    color = COLORS.Green if value > self.prev_mean[mode+key] \
                        else COLORS.Red
                self.prev_mean[mode+key] = value
                _str =f"\t{color}{mode:7} -- {key:20} -- {value:4f}{COLORS.END_NO_TOKEN}"
                tqdm.write(_str)

        tqdm.write(" ")
        tqdm.write(" ")

        self.meter = defaultdict(GroupMeters)

if __name__ == "__main__":
    import numpy as np
    N = 10
    line = CmdLineViz()
    increase = np.arange(N) + np.random.randn(N)
    decrease = -np.arange(N) + np.random.randn(N)

    for cur_inc, cur_dec in zip(increase, decrease):
        line.update("test", {
            "increase": cur_inc,
            "decrease": cur_dec
        })
    
    line.flush()

    for cur_inc, cur_dec in zip(decrease, increase):
        line.update("test", {
            "increase": cur_inc,
            "decrease": cur_dec
        })
    
    line.flush()
