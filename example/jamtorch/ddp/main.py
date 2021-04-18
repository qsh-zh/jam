from jammy.logging import Wandb
import torch.nn.functional as F
import hydra
import torch
from omegaconf import OmegaConf
import jamtorch.ddp.ddp_utils as ddp_utils
from jamtorch.ddp.ddp_trainer import DDPTrainer
from jamtorch.trainer.amp import fp16_wrapper
import jammy.utils.hyd as hyd
import torch.multiprocessing as mp
# from torch.multiprocessing import Process

@fp16_wrapper
class Trainer(DDPTrainer):
    pass

def loss_wrapper(device):
    def loss_fn(model, feed_dict, is_train):
        data, label = feed_dict
        data, label = data.to(device), label.to(device)
        pred = model(data)
        loss = F.cross_entropy(pred, label)
        return loss, {}, {}
    return loss_fn


@ddp_utils.ddp_runner
def run(cfg):
    train_set, val_set = hyd.hyd_instantiate(cfg.data.datasets)()
    train_loader, train_sampler, val_loader, val_sampler =\
        hyd.hyd_instantiate(cfg.data.dataloader, train_set, val_set, \
            rank=cfg.trainer.rank, world_size=cfg.trainer.world_size)()
    model = hyd.hyd_instantiate(cfg.model)
    device = torch.device(f"cuda:{cfg.trainer.gpu}")
    trainer = Trainer(cfg.trainer, loss_wrapper(device))
    optimizer = hyd.hyd_instantiate(cfg.optimizer, model.parameters())
    trainer.set_model_optim(model, optimizer)
    trainer.set_dataloader(train_loader, val_loader)
    trainer.set_sampler(train_sampler, val_sampler)
    trainer.train()

@ddp_utils.ddp_runner
def debug_random(cfg):
    print(cfg.trainer.gpu, torch.randn(size=(3,)))
    torch.distributed.barrier()


@hydra.main(config_path="conf/", config_name="config.yaml")
def main(cfg):
    # Wandb.launch(cfg, cfg.trainer.wandb.log, True)
    OmegaConf.set_struct(cfg, False)
    world_size = torch.cuda.device_count()
    # world_size = 2
    # mp.spawn(run, args=(world_size, None, cfg), nprocs=world_size, join=True)
    mp.spawn(debug_random, args=(world_size, None, cfg), nprocs=world_size, join=True)
    # run(cfg)
    # Wandb.finish()


if __name__ == "__main__":
    main()
