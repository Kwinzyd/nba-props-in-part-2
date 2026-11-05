import glob
import logging
import os
import sys
import time
from datetime import date
from src.config import LOG_LEVEL

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
_LOG_KEEP_DAYS = 14
_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)
    return logger


class _DailyFileHandler(logging.FileHandler):
    """
    Writes to logs/bot-YYYY-MM-DD.log and switches file at midnight. No file
    renames (unlike RotatingFileHandler), so it is safe on Windows while other
    processes have the log open.
    """
    def __init__(self, log_dir: str):
        self._dir = log_dir
        self._day = date.today().isoformat()
        super().__init__(self._path(), encoding='utf-8', delay=True)

    def _path(self) -> str:
        return os.path.join(self._dir, f'bot-{self._day}.log')

    def emit(self, record):
        day = date.today().isoformat()
        if day != self._day:
            if self.stream:
                self.stream.close()
                self.stream = None
            self._day = day
            self.baseFilename = os.path.abspath(self._path())
        super().emit(record)
        self.flush()


def enable_file_logging() -> str:
    """
    Also write every logger's output to logs/bot-YYYY-MM-DD.log (via the root
    logger, which named loggers propagate to), whatever terminal or wrapper the
    bot is started from. Deletes daily logs older than _LOG_KEEP_DAYS.
    Returns the log directory.
    """
    os.makedirs(_LOG_DIR, exist_ok=True)
    cutoff = time.time() - _LOG_KEEP_DAYS * 86400
    for old in glob.glob(os.path.join(_LOG_DIR, 'bot-*.log')):
        try:
            if os.path.getmtime(old) < cutoff:
                os.remove(old)
        except OSError:
            pass
    root = logging.getLogger()
    if not any(isinstance(h, _DailyFileHandler) for h in root.handlers):
        handler = _DailyFileHandler(_LOG_DIR)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
    return _LOG_DIR
