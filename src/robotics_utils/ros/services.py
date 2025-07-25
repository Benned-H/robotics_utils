"""Define utilities to simplify calling ROS services."""

from __future__ import annotations

from typing import Generic, TypeVar

import rospy
from std_srvs.srv import Trigger, TriggerResponse

RequestT = TypeVar("RequestT")
"""Service request message type (e.g., `AddTwoIntsRequest`)."""

ResponseT = TypeVar("ResponseT")
"""Service response message type (e.g., `AddTwoIntsResponse`)."""


class ServiceCaller(Generic[RequestT, ResponseT]):
    """Utility class to simplify ROS service calls.

    Provides a generic interface for calling a ROS service that waits for
    the service to be available and calls the service with a request.
    """

    def __init__(self, service_name: str, service_type: type, timeout_s: float = 30.0):
        """Initialize the ServiceCaller by waiting for the service to become available.

        :param service_name: Name of the ROS service to be called
        :param service_type: Message type used by the service
        :param timeout_s: Maximum duration (seconds) to wait for the service (defaults to 30)
        :raises rospy.ROSException: If the service is not available within the specified timeout
        """
        rospy.wait_for_service(service_name, timeout=timeout_s)
        self.service_name = service_name
        self._service_proxy = rospy.ServiceProxy(service_name, service_type)

    def call_service(self, request: RequestT) -> ResponseT | None:
        """Call the ROS service with the provided request message.

        :param request: Service request message
        :return: Response from the service, or None if the call fails
        """
        try:
            response: ResponseT = self._service_proxy(request)
        except rospy.ServiceException as exc:
            rospy.logerr(f"[{self.service_name}] Could not call service: {exc}")
            return None

        if response is None:
            rospy.logerr(f"[{self.service_name}] Response message was None.")
        return response

    def __call__(self, request: RequestT) -> ResponseT | None:
        """Allow the service to be called using the () operator.

        :param request: Service request message
        :return: Response from the service, or None if the call fails
        """
        return self.call_service(request)


def trigger_service(service_name: str) -> bool:
    """Call the named ROS service of the std_srvs/Trigger type.

    :param service_name: Name of the std_srvs/Trigger service to be called
    :return: Boolean success returned by the service (True or False)
    """
    rospy.wait_for_service(service_name)
    service_caller = rospy.ServiceProxy(service_name, Trigger)

    success = False
    try:
        response: TriggerResponse = service_caller()
        rospy.loginfo(f"[{service_name}] Service response message: {response.message}")
        success = response.success
    except rospy.ServiceException as exc:
        rospy.logerr(f"[{service_name}] Could not communicate with service: {exc}")

    return success
