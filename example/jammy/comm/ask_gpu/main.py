import os

import hydra

from jammy.cli_scripts.gpu_sc import get_gpu_by_utils


@hydra.main(config_path=".", config_name="config")
def my_app(cfg):
    del cfg
    pid = os.getpid()
    gpus = get_gpu_by_utils(sleep_sec=5)
    print(f"{pid}\t get {gpus}")


if __name__ == "__main__":
    my_app()  # pylint: disable=no-value-for-parameter
