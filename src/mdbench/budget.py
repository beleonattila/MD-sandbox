"""Wall-clock budget and clean-stop signalling for pausable runs.

``mdbench run CONFIG --minutes N`` starts a budget; long loops in the engines poll
:func:`stop_requested` between chunks of work, write a restart point and raise
:class:`RunPaused`. SIGTERM and SIGINT (Ctrl-C) request the same clean stop, so a service
manager or a keypress never loses more than one chunk. A second SIGINT raises
``KeyboardInterrupt`` as usual.

The budget is inactive unless :func:`start` is called (tests, ordinary runs), so engines
behave exactly as before when nobody asked for a time limit.
"""

from __future__ import annotations

import contextlib
import signal
import threading
import time
from types import FrameType


class RunPaused(Exception):
    """The run stopped at a restart point because its time budget ran out (or was signalled)."""


_deadline: float | None = None  # time.monotonic() value
_grace_s: float = 0.0
_signalled: bool = False
_active: bool = False
_old_handlers: dict[int, object] = {}


def start(minutes: float | None, grace_minutes: float = 0.0) -> None:
    """Activate the budget. ``minutes=None`` still honours signals but has no deadline."""
    global _deadline, _grace_s, _signalled, _active
    _signalled = False
    _active = True
    _grace_s = max(0.0, float(grace_minutes)) * 60.0
    _deadline = None if minutes is None else time.monotonic() + float(minutes) * 60.0


def stop() -> None:
    """Deactivate the budget and restore the signal handlers."""
    global _deadline, _active, _signalled
    _deadline = None
    _active = False
    _signalled = False
    for sig, old in _old_handlers.items():
        with contextlib.suppress(ValueError, TypeError):  # not the main thread / foreign handler
            signal.signal(sig, old)  # type: ignore[arg-type]
    _old_handlers.clear()


def active() -> bool:
    """True between :func:`start` and :func:`stop`."""
    return _active


def _on_signal(signum: int, frame: FrameType | None) -> None:
    global _signalled
    if _signalled:  # second press: give up waiting for a chunk boundary
        raise KeyboardInterrupt
    _signalled = True


def install_signal_handlers() -> None:
    """Route SIGTERM/SIGINT to a clean-stop request (main thread only)."""
    if threading.current_thread() is not threading.main_thread():
        return
    for sig in (signal.SIGTERM, signal.SIGINT):
        _old_handlers[sig] = signal.signal(sig, _on_signal)


def signalled() -> bool:
    """True once SIGTERM/SIGINT has asked for a stop."""
    return _active and _signalled


def seconds_left() -> float | None:
    """Seconds until the deadline (negative once past it); ``None`` without a deadline."""
    if not _active or _deadline is None:
        return None
    return _deadline - time.monotonic()


def stop_requested() -> bool:
    """True when the engine should stop at the next restart point it reaches."""
    if not _active:
        return False
    if _signalled:
        return True
    left = seconds_left()
    return left is not None and left <= 0.0


def grace_left() -> float:
    """Seconds an engine may still run past the deadline to reach a cheap restart point.

    Zero for signals (stop now), zero without a deadline, otherwise the grace period minus
    how far past the deadline we already are.
    """
    if not _active or _signalled:
        return 0.0
    left = seconds_left()
    if left is None:
        return 0.0
    return max(0.0, _grace_s + left)


__all__ = [
    "RunPaused",
    "active",
    "grace_left",
    "install_signal_handlers",
    "seconds_left",
    "signalled",
    "start",
    "stop",
    "stop_requested",
]
