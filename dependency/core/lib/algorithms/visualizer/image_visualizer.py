import abc

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
        import cv2
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
        import cv2
        import numpy as np
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

    @staticmethod
    def draw_bboxes_and_labels(frame, bboxes, labels):
        """
        Draws bounding boxes and corresponding labels on an image frame.

        Parameters:
            frame (numpy.ndarray): The image on which to draw the bounding boxes and labels, typically in BGR format.
            bboxes (list of tuples): A list of bounding box coordinates, where each bounding box is defined
                                     as (x_min, y_min, x_max, y_max) in pixel values.
            labels (list of str): A list of labels corresponding to each bounding box.

        Returns:
            numpy.ndarray: The modified frame with bounding boxes and labels drawn.

        Raises:
            ValueError: If `frame` is not a numpy array, `bboxes` is not a list of valid tuples,
                        `labels` is not a list, or if `labels` length does not match `bboxes`.
        """
        import cv2
        import numpy as np

        if not isinstance(frame, np.ndarray):
            raise ValueError("Input frame must be a numpy array.")

        if not isinstance(bboxes, list) or not all(isinstance(box, (tuple, list)) and len(box) == 4 for box in bboxes):
            raise ValueError(
                "Bounding boxes must be a list of tuples or lists with four numeric values (x_min, y_min, x_max, y_max)."
            )

        if not isinstance(labels, list) or len(labels) != len(bboxes):
            raise ValueError("Labels must be a list with the same length as bounding boxes.")

        for (x_min, y_min, x_max, y_max), label in zip(bboxes, labels):
            # Ensure bounding box coordinates are valid integers
            try:
                x_min, y_min, x_max, y_max = map(int, (x_min, y_min, x_max, y_max))
            except (TypeError, ValueError):
                raise ValueError("Bounding box coordinates must be convertible to integers.")

            # Check if the bounding box coordinates are within frame dimensions
            if not (0 <= x_min < x_max <= frame.shape[1]) or not (0 <= y_min < y_max <= frame.shape[0]):
                raise ValueError(
                    f"Bounding box coordinates ({x_min}, {y_min}, {x_max}, {y_max}) are out of frame bounds.")

            # Draw the bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)

            # Prepare label text
            text = str(label)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            text_color = (255, 255, 255)  # White text
            bg_color = (0, 255, 0)  # Green background

            # Calculate text size and baseline
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

            # Determine label background position
            text_bg_x1 = x_min
            text_bg_y1 = y_min - text_height - baseline - 5  # Try to place above the box

            # Adjust if the label background goes out of the frame's top
            if text_bg_y1 < 0:
                text_bg_y1 = y_max  # Place below the box if there's no space above

            text_bg_x2 = text_bg_x1 + text_width + 2
            text_bg_y2 = text_bg_y1 + text_height + baseline + 5

            # Adjust if the label background goes out of the frame's bottom
            if text_bg_y2 > frame.shape[0]:
                text_bg_y1 = y_min  # Place inside the box at the top if there's no space below

            # Draw the label background
            cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), bg_color, -1)

            # Draw the label text
            cv2.putText(
                frame,
                text,
                (text_bg_x1 + 2, text_bg_y1 + text_height + baseline // 2),
                font,
                font_scale,
                text_color,
                font_thickness,
                lineType=cv2.LINE_AA
            )

        return frame
