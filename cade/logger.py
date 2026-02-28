import logging
import logging.handlers
import os
from collections.abc import Callable
from typing import Any

from keras.callbacks import Callback


def init_log(
    log_path: str,
    level: int = logging.INFO,
    when: str = 'D',
    backup: int = 3,
    format: str = '%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s',  # noqa: E501
    datefmt: str = '%m-%d %H:%M:%S',
) -> None:
    """
    Initialize the log module.

    Args:
        log_path: Log file path prefix. Log data will go to two files:
            log_path.log and log_path.log.wf. Any non-existent parent
            directories will be created automatically.
        level: Message level threshold. DEBUG < INFO < WARNING < ERROR < CRITICAL.
        when: How to split the log file by time interval ('S', 'M', 'H', 'D', 'W').
        backup: How many backup files to keep.
        format: Format string for the log messages.
        datefmt: Format string for the date/time portion of the log.

    Raises:
        OSError: If creating log directories fails.
        IOError: If opening the log file fails.
    """
    formatter = logging.Formatter(format, datefmt)
    logger = logging.getLogger()
    logger.setLevel(level)

    dir = os.path.dirname(log_path)
    if not os.path.isdir(dir):
        os.makedirs(dir)

    handler = logging.handlers.TimedRotatingFileHandler(
        log_path + '.log', when=when, backupCount=backup
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    handler = logging.handlers.TimedRotatingFileHandler(
        log_path + '.log.wf', when=when, backupCount=backup
    )
    handler.setLevel(logging.WARNING)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class LoggingCallback(Callback):
    """Callback that logs message at end of epoch."""

    def __init__(self, print_fcn: Callable[[str], None] = print) -> None:
        super().__init__()
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if logs is None:
            logs = {}

        metrics = ', '.join(f'{k}: {v:f}' for k, v in logs.items())
        msg = f'Epoch: {epoch}, {metrics}'

        self.print_fcn(msg)
