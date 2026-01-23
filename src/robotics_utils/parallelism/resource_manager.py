"""Define a class to manage access to a limited resource across multiple threads."""

import threading
import time
from contextlib import contextmanager
from typing import Generator

from robotics_utils.io import console


class ResourceManager:
    """Manages access to a limited shared resource across multiple background threads.

    Resource priority is re-entrant: if a thread already holds priority, nested calls
    to request_priority() will succeed immediately and increment a depth counter.

    Priority is only released when the outermost context exits.
    """

    def __init__(self, grace_period_s: float = 0.0) -> None:
        """Initialize the resource manager.

        :param grace_period_s: Duration (seconds) that paused threads wait before resuming.
            If priority is re-requested within this window, other threads remain paused.
        """
        self._grace_period_s = max(0.0, grace_period_s)
        self._pause_requested = threading.Event()  # Stores a flag used to signal an event
        self._all_paused = threading.Event()

        self._registered_threads: set[str] = set()
        """Names of the threads registered with the manager."""

        self._paused_threads: set[str] = set()
        """Names of registered threads that are currently paused."""

        self._lock = threading.Lock()

        self._priority_holder: int | None = None
        """Thread ID of the thread currently holding priority (None if no holder)."""

        self._priority_depth: int = 0
        """Nesting depth for re-entrant priority calls."""

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

    def wait_for_resume(self) -> None:
        """Wait until it's safe to resume, respecting the grace period.

        Background threads should call this after acknowledge_pause() and before resuming work.
        If priority is re-requested during the grace period, the thread should stay paused.
        """
        while True:
            while self.should_pause:  # Wait for any pause request to be cleared
                time.sleep(0.05)

            if self._pause_requested.wait(timeout=self._grace_period_s):
                continue  # Priority was re-requested during the grace period; stay paused

            break  # Otherwise, we can safely resume work in the background thread

    def request_priority(self, timeout_s: float = 5.0) -> bool:
        """Request priority access to the shared resource, waiting for other threads to pause.

        This method is re-entrant: if the calling thread already holds priority, the call
        succeeds immediately and increments the nesting depth.

        :param timeout_s: Maximum duration (seconds) to wait for other threads to pause
        :return: True if all threads paused within the timeout, else False
        """
        current_thread = threading.current_thread().ident
        if current_thread is None:
            raise RuntimeError("Cannot request priority from an unstarted thread.")

        with self._lock:
            # Re-entrancy: if current thread already holds priority, increment depth
            if self._priority_holder == current_thread:
                self._priority_depth += 1
                return True

            if not self._registered_threads:  # No other threads to wait for
                self._priority_holder = current_thread
                self._priority_depth = 1
                return True

            # Check if all threads are already paused (e.g., still in grace period)
            if self._paused_threads >= self._registered_threads:
                self._pause_requested.set()
                self._priority_holder = current_thread
                self._priority_depth = 1
                return True

            self._all_paused.clear()

        self._pause_requested.set()

        # Wait for all other threads to acknowledge the pause
        all_paused = self._all_paused.wait(timeout=timeout_s)

        if all_paused:
            with self._lock:
                self._priority_holder = current_thread
                self._priority_depth = 1
        else:
            with self._lock:
                pending = self._registered_threads - self._paused_threads
                if pending:
                    console.print(f"Timeout while waiting for threads to pause: {pending}")

                # If no one holds priority, clear the pause request so threads can resume
                if self._priority_holder is None:
                    self._pause_requested.clear()

        return all_paused

    def release_priority(self) -> None:
        """Release priority access, allowing other threads to resume.

        For re-entrant calls, this decrements the nesting depth. Priority is only
        fully released when the outermost caller releases.
        """
        current_thread = threading.current_thread().ident
        if current_thread is None:
            raise RuntimeError("Cannot release priority from an unstarted thread.")

        with self._lock:
            if self._priority_holder != current_thread:  # Only the priority holder can release
                console.print(
                    f"Warning: release_priority called by non-holder thread {current_thread}.",
                )
                return

            self._priority_depth -= 1

            if self._priority_depth > 0:  # Only fully release when depth reaches 0
                return

            self._priority_holder = None

            # Note: Don't clear _paused_threads here, because threads manage their
            # own membership via acknowledge_resume(), which they call only after
            # exiting wait_for_resume() (i.e., after the grace period is over).

        self._pause_requested.clear()
        console.print("Priority released; other threads may resume.")

    @contextmanager
    def priority(self, timeout_s: float = 5.0) -> Generator[bool, None, None]:
        """Manage a context in which the caller thread has priority access to the resource.

        :param timeout_s: Maximum duration (seconds) to wait for other threads to pause
        :yield: True if all threads paused successfully, False if timeout
        """
        success = False
        try:
            success = self.request_priority(timeout_s)
            yield success
        finally:
            if success:
                self.release_priority()

    def get_status(self) -> dict:
        """Retrieve the current resource manager status for debugging.

        :return: Dictionary with registered threads, paused threads, pause state, and priority info
        """
        with self._lock:
            return {
                "registered_threads": list(self._registered_threads),
                "paused_threads": list(self._paused_threads),
                "pause_requested": self._pause_requested.is_set(),
                "priority_holder": self._priority_holder,
                "priority_depth": self._priority_depth,
            }
