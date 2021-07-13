# Consider thread safe and singleton pattern

__all__ = ["BufferCnt", "CBCnt", "ExitCnt"]


class BufferCnt:
    def __init__(self, thres=float("inf")):
        self._cnt = 0
        self.thres = thres

    def __call__(self, expre, thres=None):
        if expre is True:
            self._cnt += 1
        else:
            self._cnt = 0

        if thres is None:
            thres = self.thres

        if self._cnt > self.thres:
            self.reset()
            return True

        return False

    @property
    def cnt(self):
        return self._cnt

    def reset(self):
        self._cnt = 0


class CBCnt(BufferCnt):
    def __init__(self, cb_fn=None, thres=float("inf")):
        super().__init__(thres)
        self.call_back = cb_fn

    def __call__(self, expre, thres=None, *args, **kwargs):
        is_thres = super().__call__(expre, thres)
        if is_thres and self.call_back is not None:
            self.call_back(*args, **kwargs)


class ExitCnt(CBCnt):
    def __init__(self, msg="Something wrong", thres=float("inf")):
        def _exit_fn():
            print(msg)
            import sys

            sys.exit()

        super().__init__(_exit_fn, thres)
