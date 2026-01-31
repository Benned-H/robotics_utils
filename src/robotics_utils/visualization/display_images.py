"""Define a function to visualize images using OpenCV windows."""

from __future__ import annotations

import tkinter as tk
from functools import lru_cache
from os import environ
from pathlib import Path
from sys import base_prefix
from typing import TYPE_CHECKING, Protocol

import cv2

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@lru_cache(maxsize=None)
def find_screen_resolution() -> tuple[int, int]:
    """Find the resolution (H, W) of the current screen.

    Note: Results are cached to avoid repeated tkinter root creation/destruction,
    which can corrupt the X11 display state and break subsequent OpenCV windows.

    Reference for fixing tkinter installation issues:
        https://github.com/astral-sh/uv/issues/7036#issuecomment-2440145724
    """
    if not ("TCL_LIBRARY" in environ and "TK_LIBRARY" in environ):
        try:
            root = tk.Tk()
            root.destroy()
        except tk.TclError:
            tk_path = Path(base_prefix) / "lib"
            environ["TCL_LIBRARY"] = str(next(tk_path.glob("tcl8.*")))
            environ["TK_LIBRARY"] = str(next(tk_path.glob("tk8.*")))

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


def display_in_window(image: Displayable, title: str, *, wait: bool = True) -> bool:
    """Display an image in an OpenCV window with the given title.

    :param image: Image supporting conversion into a displayable format
    :param title: Title used for the display window (e.g., "Object Detections")
    :param wait: Whether to display the image until user input (defaults to True)
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

    full_title = f"{title} (press any key to exit)" if wait else title

    cv2.namedWindow(full_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(full_title, new_w, new_h)
    cv2.imshow(full_title, display_data)

    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return False

    key_input = cv2.waitKey(1) & 0xFF
    if key_input == ord("q"):  # Close the window if the user inputs 'q'
        cv2.destroyAllWindows()
        return False

    return True
