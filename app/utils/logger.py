import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a module logger with a simple shared format."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
        )
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger
