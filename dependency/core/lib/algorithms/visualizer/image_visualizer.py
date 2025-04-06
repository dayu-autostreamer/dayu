import abc
import cv2
import numpy as np

from core.lib.content import Task

from .base_visualizer import BaseVisualizer


class ImageVisualizer(BaseVisualizer, abc.ABC):
    default_visualization_image = 'default_visualization.png'


    def __call__(self, task: Task):
        raise NotImplementedError

    @staticmethod
    def get_first_frame_from_video(video_path):
        """
        Extracts and returns the first frame from a video file.

        Parameters:
            video_path (str): The file path to the video.

        Returns:
            numpy.ndarray: The first frame of the video in BGR format (as read by OpenCV),
                           or None if the video could not be read.

        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If the video cannot be opened or the first frame cannot be read.
        """
        # Check if the file exists
        if not isinstance(video_path, str) or not video_path:
            raise ValueError("The video path must be a valid, non-empty string.")

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        # Read the first frame
        success, frame = cap.read()
        cap.release()  # Release the video file

        if not success:
            raise ValueError("Failed to read the first frame from the video.")

        return frame

    @staticmethod
    def draw_bboxes(frame, bboxes):
        """
        Draws bounding boxes on an image frame.

        Parameters:
            frame (numpy.ndarray): The image on which to draw the bounding boxes, typically in BGR format.
            bboxes (list of tuples): A list of bounding box coordinates, where each bounding box is defined
                                     as (x_min, y_min, x_max, y_max) in pixel values.

        Returns:
            numpy.ndarray: The modified frame with bounding boxes drawn.

        Raises:
            ValueError: If `frame` is not a numpy array or if `bboxes` is not a list of valid tuples.
        """
        if not isinstance(frame, np.ndarray):
            raise ValueError("Input frame must be a numpy array.")

        if not isinstance(bboxes, list) or not all(isinstance(box, (tuple, list)) and len(box) == 4 for box in bboxes):
            raise ValueError(
                "Bounding boxes must be a list of tuples or a list of list "
                "with four numeric values (x_min, y_min, x_max, y_max).")

        for (x_min, y_min, x_max, y_max) in bboxes:
            # Ensure bounding box coordinates are valid integers
            try:
                x_min, y_min, x_max, y_max = map(int, (x_min, y_min, x_max, y_max))
            except (TypeError, ValueError):
                raise ValueError("Bounding box coordinates must be convertible to integers.")

            # Check if the bounding box coordinates are within frame dimensions
            if not (0 <= x_min < x_max <= frame.shape[1]) or not (0 <= y_min < y_max <= frame.shape[0]):
                raise ValueError(
                    f"Bounding box coordinates ({x_min}, {y_min}, {x_max}, {y_max}) are out of frame bounds.")

            # Draw the rectangle on the frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)

        return frame
