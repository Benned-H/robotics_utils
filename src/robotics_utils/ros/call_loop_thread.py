"""Define a class that continually calls a given function in an endless loop."""

from __future__ import annotations

import threading
from typing import Callable

import rospy

from robotics_utils.parallelism import ResourceManager


class CallLoopThread:
    """A class that continually calls a given function in an endless loop."""

    def __init__(
        self,
        func: Callable,
        loop_hz: float = 10.0,
        name: str | None = None,
        resource_manager: ResourceManager | None = None,
    ) -> None:
        """Initialize a thread to call the given function in a loop.

        :param func: Function to be called by the thread
        :param loop_hz: Frequency (Hz) at which the function will be called (default: 10 Hz)
        :param name: Optional identifier for the thread (default: None)
        :param resource_manager: Optional resource manager context for the thread (default: None)
        """
        self._function = func
        self._loop_hz = loop_hz
        self._name = name
        self._resource_manager = resource_manager

        if self._resource_manager is not None:
            if self._name is None:
                raise RuntimeError("Cannot register an unnamed thread with a ResourceManager.")

            self._resource_manager.register_thread(self._name)

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        """Continually call the stored function."""
        try:
            rate_hz = rospy.Rate(self._loop_hz)
            while not rospy.is_shutdown():
                if self._resource_manager is not None and self._resource_manager.should_pause:
                    self._resource_manager.acknowledge_pause(self._name)
                    while self._resource_manager.should_pause and not rospy.is_shutdown():
                        rospy.sleep(0.05)
                    self._resource_manager.acknowledge_resume(self._name)

                self._function()
                rate_hz.sleep()
        except rospy.ROSInterruptException as ros_exc:
            rospy.logwarn(f"[CallLoopThread] {ros_exc}")
        finally:
            if self._resource_manager is not None:
                self._resource_manager.unregister_thread(self._name)
