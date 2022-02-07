import pickle
from pathlib import Path

__all__ = ["AttrSaver"]


class AttrSaver:
    """
    class ABCSaver(AttrSaver):
        def __init__(self, cfg):
            super().__init__()
            self.save_path = cfg.path
            self._save_name = f"{cfg.name}_{cfg.seed:05d}"
            self.metric = {
                "metric_1": list(),
                "metric_2": dict()
            }
    """

    def __init__(self):
        self._save_name = "Foo"
        self._save_path = Path(".")

    @property
    def save_path(self):
        return self._save_path

    @save_path.setter
    def save_path(self, save_path):
        self._save_path = Path(save_path)
        self._save_path.mkdir(parents=True, exist_ok=True)

    def save(self):
        _file = f"{self._save_path}/{self._save_name}.pkl"
        with open(_file, "wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, str_path):
        with open(str_path, "rb") as handle:
            learner = pickle.load(handle)
        return learner
