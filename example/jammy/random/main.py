import hydra
from omegaconf import OmegaConf
import jammy.utils.hyd as hyd
import torch.multiprocessing as mp
from jamtorch import ddp as ddp

from jammy.random import reset_global_seed

def run(rank, world_size, unknown, cfg):
    # reset_global_seed(None, True)
    fn = hyd.hyd_instantiate(cfg.fn)
    fn()



@hydra.main(config_name="config.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    world_size = 2
    mp.spawn(run, args=(world_size, None, cfg), nprocs=world_size, join=True)



if __name__ == "__main__":
    main()
