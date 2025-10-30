"""Define a function to visualize images using OpenCV windows."""

from __future__ import annotations

import tkinter as tk
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray


def find_screen_resolution() -> tuple[int, int]:
    """Find the resolution (H, W) of the current screen."""
    root = tk.Tk()
    h_px = root.winfo_screenheight()
    w_px = root.winfo_screenwidth()
    root.destroy()
    return (h_px, w_px)


class Displayable(Protocol):
    """Protocol for images supporting visualization in OpenCV."""

    def convert_for_visualization(self) -> NDArray[np.uint8]:
        """Convert the Displayable into a form that can be visualized."""
        ...


def display_in_window(image: Displayable, window_title: str, wait_for_input: bool = True) -> bool:
    """Display an image in an OpenCV window with the given title.

    :param image: Image supporting conversion into a displayable format
    :param window_title: Title used for the display window (e.g., "Object Detections")
    :param wait_for_input: Whether to display the image until user input (defaults to True)
    :return: Boolean indicating if the window remains active (True = Active, False = Closed)
    """
    display_data = image.convert_for_visualization()

    h, w = display_data.shape[:2]
    scr_h, scr_w = find_screen_resolution()

    # If necessary, scale the image so that it fits on-screen
    if h > scr_h or w > scr_w:
        scale = min(scr_h / h, scr_w / w, 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
    else:
        new_h, new_w = h, w

    title = f"{window_title} (press any key to exit)" if wait_for_input else window_title

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, new_w, new_h)
    cv2.imshow(title, display_data)

    if wait_for_input:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return False

    key_input = cv2.waitKey(1) & 0xFF
    if key_input == ord("q"):  # Close the window if the user inputs 'q'
        cv2.destroyAllWindows()
        return False

    return True
