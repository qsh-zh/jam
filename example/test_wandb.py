from jammy.logging import Wandb
from omegaconf import OmegaConf

cfg = OmegaConf.load("pixel.yaml")

cfg.wandb.name = "test"
Wandb.launch(cfg, True)
