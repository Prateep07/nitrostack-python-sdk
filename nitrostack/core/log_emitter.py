"""Process-wide bridge from Python logging to tooling subscribers."""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable


DEFAULT_BUFFER_SIZE = 256
_STANDARD_RECORD_FIELDS = frozenset(
    logging.makeLogRecord({}).__dict__.keys()
) | {"message", "asctime"}


@dataclass(frozen=True)
class LogRecord:
    """A structured logging record delivered to tooling subscribers."""

    level: str
    levelno: int
    message: str
    timestamp: str
    logger_name: str
    extra: dict[str, Any]


class _Subscriber:
    def __init__(self, callback: Callable[[LogRecord], Any], max_buffer: int):
        self.callback = callback
        self.queue: queue.Queue[LogRecord | None] = queue.Queue(maxsize=max_buffer)
        self.active = True
        self.condition = threading.Condition()
        self.callback_thread: int | None = None
        self.thread = threading.Thread(target=self._deliver, daemon=True, name="nitrostack-log-subscriber")
        self.thread.start()

    def enqueue(self, record: LogRecord) -> None:
        with self.condition:
            if not self.active:
                return
            try:
                self.queue.put_nowait(record)
            except queue.Full:
                # Preserve the newest logs when a subscriber cannot keep up.
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.queue.put_nowait(record)
                except queue.Full:
                    pass

    def unsubscribe(self) -> None:
        with self.condition:
            if not self.active:
                return
            self.active = False
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
            self.queue.put_nowait(None)
            if self.callback_thread != threading.get_ident():
                while self.callback_thread is not None:
                    self.condition.wait()

    def _deliver(self) -> None:
        while True:
            record = self.queue.get()
            if record is None:
                return
            with self.condition:
                if not self.active:
                    continue
                self.callback_thread = threading.get_ident()
            try:
                self.callback(record)
            except Exception:
                # Subscriber code must not interfere with logging or peers.
                continue
            finally:
                with self.condition:
                    self.callback_thread = None
                    self.condition.notify_all()


class LogEmitter:
    """Thread-safe subscription point for records from ``nitrostack`` loggers."""

    def __init__(self, max_buffer: int = DEFAULT_BUFFER_SIZE):
        if max_buffer < 1:
            raise ValueError("max_buffer must be at least 1")
        self._max_buffer = max_buffer
        self._subscribers: set[_Subscriber] = set()
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[LogRecord], Any]) -> Callable[[], None]:
        """Subscribe to records; return an idempotent function to stop delivery."""
        subscriber = _Subscriber(callback, self._max_buffer)
        with self._lock:
            self._subscribers.add(subscriber)

        def unsubscribe() -> None:
            with self._lock:
                self._subscribers.discard(subscriber)
            subscriber.unsubscribe()

        return unsubscribe

    def _publish(self, record: LogRecord) -> None:
        with self._lock:
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            subscriber.enqueue(record)


class LogEmitterHandler(logging.Handler):
    """Logging handler that publishes structured records to a ``LogEmitter``."""

    def __init__(self, emitter: LogEmitter):
        super().__init__(logging.NOTSET)
        self.emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            timestamp = datetime.fromtimestamp(record.created, timezone.utc).isoformat()
            extra = {
                key: value
                for key, value in record.__dict__.items()
                if key not in _STANDARD_RECORD_FIELDS and not key.startswith("_")
            }
            self.emitter._publish(
                LogRecord(
                    level=record.levelname,
                    levelno=record.levelno,
                    message=record.getMessage(),
                    timestamp=timestamp,
                    logger_name=record.name,
                    extra=extra,
                )
            )
        except Exception:
            self.handleError(record)


log_emitter = LogEmitter()


def install_log_handler(logger_name: str = "nitrostack") -> None:
    """Attach one bridge handler to a NitroStack logger, including isolated children."""
    logger = logging.getLogger(logger_name)
    if not any(getattr(handler, "_nitrostack_log_emitter", False) for handler in logger.handlers):
        handler = LogEmitterHandler(log_emitter)
        handler._nitrostack_log_emitter = True  # type: ignore[attr-defined]
        logger.addHandler(handler)


install_log_handler()
