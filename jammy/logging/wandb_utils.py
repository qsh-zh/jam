import os
import shutil
import socket

from omegaconf import OmegaConf

from jammy.utils import git

__all__ = ["Wandb", "WandbUrls"]


class WandbUrls:  # pylint: disable=too-few-public-methods
    def __init__(self, url):

        url_hash = url.split("/")[-1]
        project = url.split("/")[-3]
        entity = url.split("/")[-4]

        self.weight_url = url
        self.log_url = "https://app.wandb.ai/{}/{}/runs/{}/logs".format(
            entity, project, url_hash
        )
        self.chart_url = "https://app.wandb.ai/{}/{}/runs/{}".format(
            entity, project, url_hash
        )
        self.overview_url = "https://app.wandb.ai/{}/{}/runs/{}/overview".format(
            entity, project, url_hash
        )
        self.hydra_config_url = (
            "https://app.wandb.ai/{}/{}/runs/{}/files/hydra-config.yaml".format(
                entity, project, url_hash
            )
        )
        self.overrides_url = (
            "https://app.wandb.ai/{}/{}/runs/{}/files/overrides.yaml".format(
                entity, project, url_hash
            )
        )

    # pylint: disable=line-too-long
    def __repr__(self):
        msg = "=================================================== WANDB URLS ===================================================================\n"
        for k, v in self.__dict__.items():
            msg += "{}: {}\n".format(k.upper(), v)
        msg += "=================================================================================================================================\n"
        return msg

    def to_dict(self):
        return {k.upper(): v for k, v in self.__dict__.items()}


class Wandb:
    IS_ACTIVE = False
    IS_HYD = False
    cfg = None
    run = None

    @staticmethod
    def set_urls_to_model(model, url):
        wandb_urls = WandbUrls(url)
        model.wandb = wandb_urls

    @staticmethod
    def _set_to_wandb_args(wandb_args, cfg, name):
        var = getattr(cfg.wandb, name, None)
        if var:
            wandb_args[name] = var

    @staticmethod
    def check_repos():
        jam_sha, jam_diff = git.log_repo(__file__)
        from jammy.utils.env import jam_getenv

        if jam_getenv("proj_path"):
            proj_dir = jam_getenv("proj_path")
        else:
            import __main__ as _main

            proj_dir = _main.__file__
        project_sha, project_diff = git.log_repo(proj_dir)
        with open("jam_change.patch", "w") as f:
            f.write(jam_diff)
        with open("proj_change.patch", "w") as f:
            f.write(project_diff)

        return jam_sha, proj_dir, project_sha

    @staticmethod
    def prep_args(cfg):
        jam_sha, proj_dir, proj_sha = Wandb.check_repos()
        wandb_args = {
            "project": cfg.wandb.project,
            "resume": "allow",
            # "tags": cfg.wandb.tags,
            "config": {
                "run_path": os.getcwd(),
                "jam_sha": jam_sha,
                "proj_path": proj_dir,
                "proj_sha": proj_sha,
                "hydra": Wandb.IS_HYD,
                "host": socket.gethostname(),
            },
        }
        for key in ["name", "entity", "notes", "id", "tags"]:
            Wandb._set_to_wandb_args(wandb_args, cfg, key)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if "wandb" in cfg_dict:
            del cfg_dict["wandb"]

        wandb_args["config"]["z"] = cfg_dict

        return wandb_args

    @staticmethod
    def launch(cfg, launch: bool, is_hydra: bool = True, dump_meta: bool = True):
        Wandb.IS_HYD = is_hydra
        if launch:
            import wandb

            Wandb.IS_ACTIVE = True
            wandb_args = Wandb.prep_args(cfg)
            Wandb.run = wandb.init(**wandb_args)
            Wandb.cfg = {**wandb_args["config"], **(WandbUrls(Wandb.run.url).to_dict())}

            wandb.save(os.path.join(os.getcwd(), "jam_change.patch"))
            wandb.save(os.path.join(os.getcwd(), "proj_change.patch"))

            if is_hydra:
                shutil.copyfile(
                    os.path.join(os.getcwd(), ".hydra/config.yaml"),
                    os.path.join(os.getcwd(), ".hydra/hydra-config.yaml"),
                )
                wandb.save(os.path.join(os.getcwd(), ".hydra/hydra-config.yaml"))
                wandb.save(os.path.join(os.getcwd(), ".hydra/overrides.yaml"))
        else:
            Wandb.cfg = Wandb.prep_args(cfg)["config"]
        if dump_meta:
            with open("meta.yaml", "w") as fp:
                OmegaConf.save(config=OmegaConf.create(Wandb.cfg), f=fp.name)
        return Wandb.run

    @staticmethod
    def add_file(file_path: str):
        if not Wandb.IS_ACTIVE:
            return
        import wandb

        filename = os.path.basename(file_path)
        shutil.copyfile(file_path, os.path.join(wandb.run.dir, filename))

    @staticmethod
    def log(*args, **kargs):
        if not Wandb.IS_ACTIVE:
            raise RuntimeError("wandb is inactive, please launch first.")
        import wandb

        wandb.log(*args, **kargs)

    @staticmethod
    def finish():
        if not Wandb.IS_ACTIVE:
            return
        import wandb

        if os.path.exists("jam_.log"):
            Wandb.add_file("jam_.log")

        wandb.finish()
