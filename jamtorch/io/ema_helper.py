from jammy.utils.hyd import hyd_instantiate

from jamtorch.logging import get_logger

logger = get_logger()

__all__ = ["hyd_ema"]

def hyd_ema(cfg):
    is_ema = False
    if "enable_ema" in cfg:
        if cfg.enable_ema != True:
            return None
    if "ema" in cfg:
        if bool(cfg.ema):
            return hyd_instantiate(cfg.ema)
        if "enable_ema" in cfg and cfg.enable_ema == True:
            logger.critical("ema fail instantiate")
    return None

