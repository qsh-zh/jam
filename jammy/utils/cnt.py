# Consider thread safe and singleton pattern
import inspect
from collections import defaultdict

__all__ = ["BufferCnt", "CBCnt", "ExitCnt", "bufcnt"]

g_caller_cnt = defaultdict(lambda: 0)


def bufcnt(cond):
    frames = inspect.stack()
    hash_info = "".join(
        [f"{cur_st.filename}{cur_st.lineno}\n" for cur_st in frames[1:]]
    )
    g_caller_cnt[hash_info] = g_caller_cnt[hash_info] * cond + cond
    return g_caller_cnt[hash_info]


class BufferCnt:
    def __init__(self, thres=float("inf"), reset_over_thres=False):
        self._cnt = 0
        self.thres = thres
        self.reset_over_thres = reset_over_thres

    def __call__(self, expre, thres=None):
        if expre is True:
            self._cnt += 1
        else:
            self._cnt = 0

        if thres is None:
            thres = self.thres

        if self._cnt > self.thres:
            if self.reset_over_thres:
                self.reset()
            return True

        return False

    @property
    def cnt(self):
        return self._cnt

    def reset(self):
        self._cnt = 0


class CBCnt(BufferCnt):
    def __init__(self, cb_fn=None, thres=float("inf"), reset_over_thres=False):
        super().__init__(thres, reset_over_thres=reset_over_thres)
        self.call_back = cb_fn

    def __call__(self, expre, thres=None, *args, **kwargs):
        is_thres = super().__call__(expre, thres)
        if is_thres and self.call_back is not None:
            self.call_back(*args, **kwargs)


class ExitCnt(CBCnt):
    def __init__(
        self, msg="Something wrong", thres=float("inf"), reset_over_thres=False
    ):
        def _exit_fn():
            print(msg)
            import sys

            sys.exit()

        super().__init__(_exit_fn, thres, reset_over_thres)
