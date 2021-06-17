try:
    from .utils import *
except ImportError:
    from jammy.logging import get_logger
    logger = get_logger()
    logger.critical("Import Error, check dependency of torch packages")



from jamtorch.utils.init import init_main

init_main()

del init_main