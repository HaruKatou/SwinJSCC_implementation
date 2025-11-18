import logging
from .helpers import makedirs

def configure_logger(config, save_log: bool = False, test_mode: bool = False) -> logging.Logger:
    """
    Configure and return a logger for training or testing.

    Args:
        config: configuration object with attributes like workdir, log, samples, models.
        save_log: whether to save logs to file.
        test_mode: whether to append '_test' to workdir for testing runs.
    """
    logger = logging.getLogger("DeepJointSourceChannelCoder")

    if logger.hasHandlers():
        logger.handlers.clear()

    if test_mode:
        config.workdir += "_test"

    # Prepare directories if logging to disk
    if save_log:
        for path in [config.workdir, config.samples, config.models]:
            makedirs(path)

    formatter = logging.Formatter("%(asctime)s - [%(levelname)s] %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if save_log:
        file_handler = logging.FileHandler(config.log)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    config.logger = logger

    return logger