"""
Ask Windows not to idle-sleep while the bot runs (like a video player does).

The laptop idle-slept through alert windows (e.g. Nov 5 5:21–6:24 PM, right
before tip-offs). SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
holds the system awake until the calling thread exits, so call it from the
long-lived main thread. The display may still turn off; closing the lid or
pressing the power button still sleeps the PC. No-op off Windows.
"""

import sys

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def keep_system_awake() -> bool:
    """Returns True if Windows accepted the request."""
    if sys.platform != 'win32':
        return False
    try:
        import ctypes
        prev = ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
    except Exception as e:
        logger.warning(f"Keep-awake unavailable: {e}")
        return False
    if not prev:
        logger.warning("Keep-awake: Windows rejected SetThreadExecutionState.")
        return False
    logger.info("Keep-awake on: Windows won't idle-sleep while the bot runs (lid/power button still sleep).")
    return True
