import logging


def get_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='training.log', encoding='utf-8', level=logging.INFO)
    return logger