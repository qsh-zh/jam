from jamtorch.utils.meta import is_master
from jamtorch.io import hyd_ema

__all__ = ["setup_ema"]

def setup_ema(cfg, trainer):
    trainer.ema = None
    # FIXME: check ema_master key word design
    if cfg.ema_master: # only one ema on the master
        if is_master():
            trainer.ema = hyd_ema(cfg)
    else:
        trainer.ema = hyd_ema(cfg)

    if trainer.ema:
        def update_ema(_trainer):
            trainer.ema.update_parameters(trainer.model)
        
        trainer.register_event("step:end", update_ema, False)


