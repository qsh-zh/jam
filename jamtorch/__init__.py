from jammy.utils.env import jam_getenv

try:
    from .utils import *
except ImportError:
    from jammy.logging import get_logger

    logger = get_logger()
    logger.critical("Import Error, check dependency of torch packages")


from jamtorch.utils.init import init_main

init_main()

del init_main

if jam_getenv("IMPORT_ALL", "true", "bool"):
    from jamtorch.cuda import set_best_device
    from jamtorch.logging import get_logger  # pylint: disable=ungrouped-imports
