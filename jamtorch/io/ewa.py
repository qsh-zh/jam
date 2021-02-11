import copy

__all__ = ["EWA"]


# FIXME: parallel models, need to deal with model


class EWA:
    def __init__(self, beta, num_warm, num_every, model=None):
        self.beta = beta
        self.num_warm = num_warm
        self.num_every = num_every
        self.cnt = 0
        self._old_model = copy.deepcopy(model)

    def update_model_average(self, new_model):
        self.cnt += 1
        if self.cnt < self.num_warm:
            return
        if self.model is None:
            self.model = new_model
        if self.cnt = self.num_warm:
            self._old_model.load_state_dict(new_model.state_dict())
            return
        if self.cnt % self.num_every == 0:
            for old_param, new_param in zip(
                self.model.parameters(), new_model.parameters()
            ):
                old_weight, new_weight = old_param.data, new_param.data
                old_param.data = old_weight * self.beta + (1 - self.beta) * new_weight

    @property
    def model(self):
        return self._old_model

    @model.setter
    def model(self, model):
        self._old_model = copy.deepcopy(model)

    def dump2dict(self):
        return {
            "cnt": self.cnt,
            "model": self._old_model.state_dict(),
            "num_warm": self.num_warm,
            "beta": self.beta,
        }

    def load_dict(self, ewa_dict):
        self.cnt = ewa_dict.get("cnt") or 0
        self.num_warm = ewa_dict.get("num_warm") or 0
        self.model = self._old_model.load_state_dict(ewa_dict.get("model"))
        self.beta = ewa_dict.get("beta") or 0.9
