"""Define a class that continually calls a given function in an endless loop."""

import threading
from typing import Callable

import rospy


class CallLoopThread:
    """A class that continually calls a given function in an endless loop."""

    def __init__(self, func: Callable, loop_hz: float = 10.0) -> None:
        """Initialize a thread to call the given function in a loop."""
        self._function = func
        self._loop_hz = loop_hz
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        """Continually call the stored function."""
        try:
            rate_hz = rospy.Rate(self._loop_hz)
            while not rospy.is_shutdown():
                self._function()
                rate_hz.sleep()
        except rospy.ROSInterruptException as ros_exc:
            rospy.logwarn(f"[CallLoopThread] {ros_exc}")
