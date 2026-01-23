"""Define a class to manage access to a limited resource across multiple threads."""

import threading
from contextlib import contextmanager
from typing import Generator

from robotics_utils.io import console


class ResourceManager:
    """Manages access to a limited shared resource across multiple background threads."""

    def __init__(self) -> None:
        """Initialize the resource manager."""
        self._pause_requested = threading.Event()  # Stores a flag used to signal an event
        self._all_paused = threading.Event()

        self._registered_threads: set[str] = set()
        """Names of the threads registered with the manager."""

        self._paused_threads: set[str] = set()
        """Names of registered threads that are currently paused."""

        self._lock = threading.Lock()

    @property
    def should_pause(self) -> bool:
        """Check if the background threads should pause (non-blocking).

        :return: True if a priority operation has requested a pause
        """
        return self._pause_requested.is_set()

    def _check_all_paused(self) -> None:
        """Check if all registered threads have been paused (caller must hold the lock)."""
        if self._paused_threads >= self._registered_threads:
            self._all_paused.set()
        else:
            self._all_paused.clear()

    def register_thread(self, name: str) -> None:
        """Register a background thread that uses the shared resource."""
        with self._lock:
            self._registered_threads.add(name)
            console.print(f"Registered thread: {name}")

    def unregister_thread(self, name: str) -> None:
        """Unregister a background thread."""
        with self._lock:
            self._registered_threads.discard(name)
            self._paused_threads.discard(name)
            self._check_all_paused()
            console.print(f"Unregistered thread: {name}")

    def acknowledge_pause(self, name: str) -> None:
        """Acknowledge that the named caller thread has paused.

        :param name: Identifier of the calling thread acknowledging the pause
        """
        with self._lock:
            if name in self._registered_threads:
                self._paused_threads.add(name)
                console.print(f"Thread paused: {name}")
                self._check_all_paused()

    def acknowledge_resume(self, name: str) -> None:
        """Acknowledge that the named caller thread has resumed.

        :param name: Identifier of the calling thread that has resumed
        """
        with self._lock:
            self._paused_threads.discard(name)
            console.print(f"Thread resumed: {name}")

    def request_priority(self, timeout_s: float = 5.0) -> bool:
        """Request priority access to the shared resource, waiting for other threads to pause.

        :param timeout_s: Maximum duration (seconds) to wait for other threads to pause
        :return: True if all threads paused within the timeout, else False
        """
        with self._lock:
            if not self._registered_threads:  # No other threads to wait for
                return True
            self._all_paused.clear()

        self._pause_requested.set()

        # Wait for all other threads to acknowledge the pause
        all_paused = self._all_paused.wait(timeout=timeout_s)

        if not all_paused:
            with self._lock:
                pending = self._registered_threads - self._paused_threads
                if pending:
                    console.print(f"Timeout while waiting for threads to pause: {pending}")

        return all_paused

    def release_priority(self) -> None:
        """Release priority access, allowing other threads to resume."""
        with self._lock:
            self._paused_threads.clear()
            self._check_all_paused()

        self._pause_requested.clear()
        console.print("Priority released; other threads may resume.")

    @contextmanager
    def priority(self, timeout_s: float = 5.0) -> Generator[bool, None, None]:
        """Manage a context in which the caller thread has priority access to the resource.

        :param timeout_s: Maximum duration (seconds) to wait for other threads to pause
        :yield: True if all threads paused successfully, False if timeout
        """
        try:
            success = self.request_priority(timeout_s)
            yield success
        finally:
            self.release_priority()

    def get_status(self) -> dict:
        """Retrieve the current resource manager status for debugging.

        :return: Dictionary with registered threads, paused threads, and the pause state
        """
        with self._lock:
            return {
                "registered_threads": list(self._registered_threads),
                "paused_threads": list(self._paused_threads),
                "pause_requested": self._pause_requested.is_set(),
            }
