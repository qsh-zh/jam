# pylint: disable=invalid-name
import os
import os.path as osp
import sys
import tempfile

import yaml

from jammy.logging import get_logger

# logger = get_logger(sys.stdout, colorize=True,
#                   format="<green>{time}</green> <level>{message}</level>")
logger = get_logger(sys.stderr, colorize=True, level="INFO")


def load_system_settings(root, config, bash_file):  # pylint: disable=unused-argument
    if "system" not in config or config["system"] is None:
        return
    envs = config["system"].get("envs", {})
    for k, v in envs.items():
        logger.info("Export system environment variable {} = {}.".format(k, v))
        print("export {}={}".format(k, v), file=bash_file)


def load_vendors(root, config, bash_file):  # pylint: disable=unused-argument
    if "vendors" not in config or config["vendors"] is None:
        return

    for k, v in config["vendors"].items():
        assert "root" in v, '"root" not found in vendor: {}.'.format(k)

        logger.info("Loading vendor: {}.".format(k))
        print(
            "export PYTHONPATH={}:$PYTHONPATH".format(osp.join(root, v["root"])),
            file=bash_file,
        )


def load_conda_settings(root, config, bash_file):  # pylint: disable=unused-argument
    if "conda" not in config or config["conda"] is None:
        return
    target_env = config["conda"].get("env", "")
    if target_env != "":
        logger.info("Using conda env: {}.".format(target_env))
        print(
            """
if [[ $CONDA_DEFAULT_ENV != "{v}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate {v}
fi
""".format(
                v=target_env
            ),
            file=bash_file,
        )


def load_yml_config(root, bash_file, recursive=False):
    if recursive:
        last_root = None
        while root != last_root:
            yml_filename = osp.join(root, "jam_.yaml")
            if osp.isfile(yml_filename):
                break
            last_root = root
            root = osp.dirname(root)
    else:
        yml_filename = osp.join(root, "jam_.yaml")

    if osp.isfile(yml_filename):
        logger.critical("Loading jam config: {}.".format(osp.abspath(yml_filename)))
        with open(yml_filename) as f:
            config = yaml.safe_load(f.read())
        if config is not None:
            load_system_settings(root, config, bash_file)
            load_vendors(root, config, bash_file)
            load_conda_settings(root, config, bash_file)


def main():
    f = tempfile.NamedTemporaryFile("w", delete=False)
    load_yml_config(
        osp.dirname(osp.dirname(__file__)), f
    )  # jammy package configuration
    load_yml_config(os.getcwd(), f, recursive=True)  # project configuration
    f.close()
    print(f.name)


if __name__ == "__main__":
    main()
