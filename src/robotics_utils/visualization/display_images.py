"""Define a function to visualize images using OpenCV windows."""

from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray


class Displayable(Protocol):
    """Protocol for images supporting visualization in OpenCV."""

    def convert_for_visualization(self) -> NDArray[np.uint8]:
        """Convert the Displayable into a form that can be visualized."""


def display_image(image: Displayable, window_title: str, wait_for_input: bool = True) -> bool:
    """Display an image in an OpenCV window with the given title.

    :param image: Image supporting conversion into a displayable format
    :param window_title: Title used for the display window
    :param wait_for_input: Whether to display the image until user input (defaults to True)
    :return: Boolean indicating if the window remains active (True = Active, False = Closed)
    """
    display_data = image.convert_for_visualization()
    cv2.imshow(window_title, display_data)

    if wait_for_input:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return False

    key_input = cv2.waitKey(1) & 0xFF
    if key_input == ord("q"):  # Close the window if the user inputs 'q'
        cv2.destroyAllWindows()
        return False

    return True
