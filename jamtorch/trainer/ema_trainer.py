from .simple_trainer import SimpleTrainer

import jammy.utils.hyd as hyd

class EMATrainer(SimpleTrainer):
    def __init__(self, cfg, loss_fn):
        super().__init__(cfg, loss_fn)
        self.ema = hyd.hyd_instantiate(cfg.ema)
        if self.ema:
            def update_ema(trainer):
                trainer.ema.update_parameters(trainer.model)
            self.register_event("step:end",update_ema, False)

    def _impl_load_ckpt(self, state):
        if self.ema:
            self.ema.load_state_dict(state["ema"])
        return super()._impl_load_ckpt(state)

    def _impl_save_ckpt(self):
        rtn = {}
        if self.ema:
            rtn["ema"] = self.ema.state_dict()
        return {**super()._impl_save_ckpt(), **rtn}