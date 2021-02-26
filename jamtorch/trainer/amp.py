try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

import torch

from .genetic_trainer import GeneticTrainer


__all__ = ["loss_backwards", "fp16_wrapper"]


def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


def amp_init(trainer, *args, **kwargs):
    trainer.model, trainer.optimizer = amp.initialize(
        trainer.model, trainer.optimizier, opt_level="O1"
    )


def amp_state_dict(*args, **kwargs):
    return {"amp": amp.state_dict()}


def amp_load_state(state, *args, **kwargs):
    """
    call after the initialization!!! Pay attention to the location.
    """
    amp.load_state_dict(state["amp"])

# Replace the GeneticTrainer by any trainer
class Fp16GeneticTrainer(GeneticTrainer):
    def __init__(self, cfg, loss_fn):
        super().__init__(cfg, loss_fn)
        
        self.fp16 = cfg.get("fp16") or False
        self.setup_fp16()

    def set_model_optim(self, model, optimizer):
        if self.fp16:
            model, optimizer = amp.initialize(
                    model, optimizer, opt_level="O1"
                )
        super().set_model_optim(model, optimizer=optimizer)

    def loss_backward(self, loss):
        if self.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                value = super().loss_backward(scaled_loss)
            return value
        return super().loss_backward(loss)

    def setup_fp16(self):
        def _fp16_load(trainer, state):
            amp_load_state(state)
        def _fp16_export(trainer, state_dict):
            state_dict.update(amp_state_dict())
        self.register_event("trainer:load",_fp16_load)
        self.register_event("trainer:export", _fp16_export)


def fp16_wrapper(base_class):
    class _FPClass(base_class):
        def __init__(self,*args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

            self.fp16 = self._cfg.get("fp16") or False
            self.setup_fp16()

        def set_model_optim(self, model, optimizer):
            if next(model.parameters()).device == torch.device("cpu"):
                model = model.to(self.device)
            if self.fp16:
                model, optimizer = amp.initialize(
                        model, optimizer, opt_level="O1"
                    )
            super().set_model_optim(model, optimizer=optimizer)

        def loss_backward(self, loss):
            if self.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    value = super().loss_backward(scaled_loss)
                return value
            return super().loss_backward(loss)

        def setup_fp16(self):
            def _fp16_load(trainer, state):
                if trainer.fp16:
                    amp_load_state(state)
            def _fp16_export(trainer, state_dict):
                if trainer.fp16:
                    state_dict.update(amp_state_dict())
            self.register_event("trainer:load",_fp16_load)
            self.register_event("trainer:export", _fp16_export)

    return _FPClass