"""Robust camera feed component that avoids Tkinter image garbage collection issues."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageTk


class CameraFeed(tk.Canvas):
    """
    A robust camera feed component that avoids Tkinter PhotoImage garbage collection
    by using a single persistent PhotoImage object and updating it in-place.
    """

    def __init__(
        self,
        master,
        width: int = 960,
        height: int = 540,
        bg_color: str = "#0F1117",
        **kwargs
    ):
        super().__init__(
            master,
            width=width,
            height=height,
            bg=bg_color,
            highlightthickness=0,
            **kwargs
        )
        self._width = width
        self._height = height
        
        # Frame queue for thread-safe communication
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        
        # Single persistent PhotoImage to avoid garbage collection
        self._photo_image: ImageTk.PhotoImage | None = None
        self._current_image_id: str | None = None
        
        # Frame buffer for smooth display
        self._frame_buffer: deque = deque(maxlen=3)
        self._last_frame_time: float = 0
        self._frame_count: int = 0
        
        # State tracking
        self._is_running: bool = False
        self._update_job: str | None = None
        self._no_camera_placeholder = self._create_no_camera_placeholder()
        
        # Display the placeholder initially
        self._display_placeholder()

    def _create_no_camera_placeholder(self) -> np.ndarray:
        """Create a 'No Camera' placeholder image."""
        placeholder = np.full((self._height, self._width, 3), (15, 23, 42), dtype=np.uint8)
        return placeholder

    def _display_placeholder(self) -> None:
        """Display the 'No Camera' placeholder."""
        try:
            pil_image = Image.fromarray(self._no_camera_placeholder)
            if self._photo_image is None:
                self._photo_image = ImageTk.PhotoImage(pil_image)
            else:
                # Update the existing PhotoImage in-place
                self._photo_image.paste(pil_image)
            
            self.delete("all")
            self._current_image_id = self.create_image(
                self._width // 2, self._height // 2,
                image=self._photo_image,
                anchor="center"
            )
        except Exception:
            pass  # Fail silently during initialization

    def update_frame(self, frame: np.ndarray) -> None:
        """
        Called from camera thread to provide a new frame.
        Uses a queue to avoid blocking the camera thread.
        """
        try:
            if not self._frame_queue.full():
                self._frame_queue.put(frame.copy())
        except Exception:
            pass  # Drop frame if queue is full

    def start_updates(self, interval_ms: int = 50) -> None:
        """Start the update loop for displaying frames."""
        if not self._is_running:
            self._is_running = True
            self._schedule_update(interval_ms)

    def stop_updates(self) -> None:
        """Stop the update loop."""
        self._is_running = False
        if self._update_job:
            self.after_cancel(self._update_job)
            self._update_job = None

    def _schedule_update(self, interval_ms: int) -> None:
        """Schedule the next update."""
        if self._is_running:
            self._update_job = self.after(interval_ms, self._update_loop)

    def _update_loop(self) -> None:
        """Main update loop that runs on the UI thread."""
        try:
            # Get the latest frame from the queue
            frame = None
            try:
                while not self._frame_queue.empty():
                    frame = self._frame_queue.get_nowait()
            except queue.Empty:
                pass

            if frame is not None and frame.size > 0:
                self._display_frame(frame)
            else:
                # No frame available, show placeholder
                self._display_placeholder()

        except Exception as e:
            print(f"[CameraFeed] Error in update loop: {e}")
        
        # Schedule next update
        self._schedule_update(50)  # 20 FPS target

    def _display_frame(self, frame: np.ndarray) -> None:
        """Display a frame using the persistent PhotoImage."""
        try:
            # Resize frame to display size
            if frame.shape[:2] != (self._height, self._width):
                frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_LINEAR)

            # Convert BGR to RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create PIL image
            pil_image = Image.fromarray(frame)

            # Create or update PhotoImage
            if self._photo_image is None:
                self._photo_image = ImageTk.PhotoImage(pil_image)
            else:
                # Update the existing PhotoImage in-place
                self._photo_image.paste(pil_image)

            # Display on canvas
            self.delete("all")
            self._current_image_id = self.create_image(
                self._width // 2, self._height // 2,
                image=self._photo_image,
                anchor="center"
            )

            self._frame_count += 1
            self._last_frame_time = time.time()

        except Exception as e:
            print(f"[CameraFeed] Error displaying frame: {e}")

    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_updates()
        self.delete("all")
        if self._photo_image:
            del self._photo_image
            self._photo_image = None