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
    def get_frame_from_video(video_path, frame_index=0):
        import cv2

        if not isinstance(video_path, str) or not video_path:
            raise ValueError("The video path must be a valid, non-empty string.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        try:
            if frame_index == 'last':
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                target_index = max(0, frame_count - 1)
            else:
                target_index = max(0, int(frame_index))
            if target_index > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
            success, frame = cap.read()
        finally:
            cap.release()

        if not success:
            raise ValueError(f"Failed to read frame {frame_index} from the video.")

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
        Draws bounding boxes and corresponding labels on an image frame with improved visibility.

        Parameters:
            frame (numpy.ndarray): The image in BGR format.
            bboxes (list of tuples): Bounding boxes in (x_min, y_min, x_max, y_max) format.
            labels (list of str): Text labels corresponding to each bounding box.

        Returns:
            numpy.ndarray: Modified frame with drawn elements.

        Raises:
            ValueError: For invalid input formats or out-of-bounds coordinates.
        """
        import cv2
        import numpy as np

        # Input validation
        if not isinstance(frame, np.ndarray):
            raise ValueError("Input frame must be a numpy array.")

        if not isinstance(bboxes, list) or not all(isinstance(box, (tuple, list)) and len(box) == 4 for box in bboxes):
            raise ValueError("Bounding boxes must be a list of 4-element tuples/lists.")

        if not isinstance(labels, list) or len(labels) != len(bboxes):
            raise ValueError("Labels must be a list with same length as bboxes.")

        for (x_min, y_min, x_max, y_max), label in zip(bboxes, labels):
            # Convert coordinates to integers
            try:
                x_min, y_min, x_max, y_max = map(int, (x_min, y_min, x_max, y_max))
            except (TypeError, ValueError):
                raise ValueError("Bounding box coordinates must be numeric.")

            # Validate coordinates
            if not (0 <= x_min < x_max <= frame.shape[1]) or not (0 <= y_min < y_max <= frame.shape[0]):
                raise ValueError(f"Invalid coordinates: ({x_min}, {y_min}, {x_max}, {y_max})")

            # Draw bounding box
            box_color = (0, 255, 0)  # Green
            box_thickness = 4
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, box_thickness)

            # Configure text parameters
            text = str(label)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0  # Increased from 0.6
            font_thickness = 2  # Increased from 1
            text_color = (0, 255, 0)  # Green text
            vertical_offset = 5  # Space between text and box

            # Calculate text position
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

            # Position text above box (with fallback to inside position)
            if y_min - text_h - vertical_offset > 0:  # Enough space above
                text_org = (x_min, y_min - vertical_offset)
            else:  # Place inside top-left corner
                text_org = (x_min, y_min + text_h + vertical_offset)

            # Ensure text doesn't go out of frame
            text_org = (
                max(0, min(text_org[0], frame.shape[1] - text_w)),  # X coordinate
                max(text_h, min(text_org[1], frame.shape[0]))  # Y coordinate
            )

            # Draw text with anti-aliasing
            cv2.putText(frame, text, text_org, font, font_scale,
                        text_color, font_thickness, cv2.LINE_AA)

        return frame

    @staticmethod
    def extract_bboxes(content, frame_index=0):
        return [
            item.get('bbox')
            for item in ImageVisualizer.extract_items(content, 'bbox', frame_index)
            if isinstance(item, dict) and len(item.get('bbox') or []) == 4
        ]

    @staticmethod
    def extract_texts(content, frame_index=0):
        return [
            str(item.get('text', item.get('label', '')))
            for item in ImageVisualizer.extract_items(content, 'text', frame_index)
        ]

    @staticmethod
    def extract_records(content, label):
        if not isinstance(content, dict):
            return []
        outputs = content.get('outputs')
        if not isinstance(outputs, dict):
            return []
        records = outputs.get(label) or []
        return [record for record in records if isinstance(record, dict)]

    @staticmethod
    def extract_items(content, label, frame_index=0, include_global=True):
        records = ImageVisualizer.extract_records(content, label)
        if frame_index == 'last':
            frame_indices = [
                record.get('frame_index') for record in records
                if isinstance(record.get('frame_index'), int)
            ]
            if frame_indices:
                frame_index = max(frame_indices)
        selected = [
            record for record in records
            if record.get('frame_index') == frame_index
            or (include_global and record.get('frame_index') is None)
        ]
        if not selected and records:
            selected = records[:1]
        items = []
        for record in selected:
            items.extend(record.get('items') or [])
        return items

    @staticmethod
    def get_nested_value(item, field):
        value = item
        for part in str(field).split('.'):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    @staticmethod
    def format_label_value(value):
        if isinstance(value, float):
            return f'{value:.2f}'
        return str(value)

    @staticmethod
    def item_label(item, fields=None, template=None, fallback_fields=None):
        if not isinstance(item, dict):
            return ''

        if template:
            label = str(template)
            for field in fields or []:
                value = ImageVisualizer.get_nested_value(item, field)
                label = label.replace('{' + str(field) + '}', '' if value is None else ImageVisualizer.format_label_value(value))
            return label

        label_parts = []
        candidate_fields = fields or fallback_fields or [
            'text',
            'label',
            'category',
            'state',
            'intent',
            'track_id',
            'person_id',
        ]
        for field in candidate_fields:
            value = ImageVisualizer.get_nested_value(item, field)
            if value is not None and value != '':
                label_parts.append(ImageVisualizer.format_label_value(value))

        if not fields:
            for score_field in ('score', 'confidence', 'prob', 'risk_score', 'abnormal_stop_prob'):
                value = ImageVisualizer.get_nested_value(item, score_field)
                if isinstance(value, (int, float)):
                    label_parts.append(f'{score_field}:{float(value):.2f}')
                    break

        return ' '.join(label_parts)

    @staticmethod
    def clip_bbox(frame, bbox):
        if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            return None
        try:
            x_min, y_min, x_max, y_max = [int(round(float(value))) for value in bbox]
        except (TypeError, ValueError):
            return None
        height, width = frame.shape[:2]
        x_min = max(0, min(width - 1, x_min))
        x_max = max(0, min(width, x_max))
        y_min = max(0, min(height - 1, y_min))
        y_max = max(0, min(height, y_max))
        if x_max <= x_min or y_max <= y_min:
            return None
        return [x_min, y_min, x_max, y_max]

    @staticmethod
    def _point(value):
        if not isinstance(value, (tuple, list)) or len(value) < 2:
            return None
        try:
            return int(round(float(value[0]))), int(round(float(value[1])))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def draw_text(frame, text, origin, color=(255, 255, 255), background=(15, 23, 42), font_scale=0.48):
        import cv2

        if not text:
            return frame
        x, y = origin
        height, width = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(str(text), font, font_scale, thickness)
        x = max(0, min(int(x), max(width - text_w - 4, 0)))
        y = max(text_h + 4, min(int(y), height - baseline - 2))
        cv2.rectangle(frame, (x, y - text_h - 4), (x + text_w + 6, y + baseline + 2), background, -1)
        cv2.putText(frame, str(text), (x + 3, y), font, font_scale, color, thickness, cv2.LINE_AA)
        return frame

    @staticmethod
    def draw_safe_bbox(frame, bbox, label='', color=(0, 255, 0), thickness=2):
        import cv2

        clipped = ImageVisualizer.clip_bbox(frame, bbox)
        if not clipped:
            return frame
        x_min, y_min, x_max, y_max = clipped
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
        if label:
            ImageVisualizer.draw_text(frame, label, (x_min, y_min - 6), color=(255, 255, 255), background=color)
        return frame

    @staticmethod
    def draw_polyline(frame, points, color=(255, 255, 0), thickness=2, closed=False):
        import cv2
        import numpy as np

        valid_points = [ImageVisualizer._point(point) for point in points or []]
        valid_points = [point for point in valid_points if point is not None]
        if len(valid_points) < 2:
            return frame
        array = np.array(valid_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [array], bool(closed), color, thickness, cv2.LINE_AA)
        return frame

    @staticmethod
    def draw_polygon(frame, points, color=(0, 180, 255), alpha=0.28, outline_thickness=2):
        import cv2
        import numpy as np

        valid_points = [ImageVisualizer._point(point) for point in points or []]
        valid_points = [point for point in valid_points if point is not None]
        if len(valid_points) < 3:
            return frame
        array = np.array(valid_points, dtype=np.int32).reshape((-1, 1, 2))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [array], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [array], True, color, outline_thickness, cv2.LINE_AA)
        return frame

    @staticmethod
    def draw_keypoints(frame, keypoints, color=(255, 0, 255), links=None):
        import cv2

        points = []
        for keypoint in keypoints or []:
            if not isinstance(keypoint, (tuple, list)):
                points.append(None)
                continue
            if len(keypoint) >= 3:
                try:
                    if float(keypoint[2]) <= 0:
                        points.append(None)
                        continue
                except (TypeError, ValueError):
                    pass
            points.append(ImageVisualizer._point(keypoint))

        if links:
            for first, second in links:
                if first < len(points) and second < len(points) and points[first] and points[second]:
                    cv2.line(frame, points[first], points[second], color, 2, cv2.LINE_AA)
        for point in points:
            if point:
                cv2.circle(frame, point, 4, color, -1, cv2.LINE_AA)
        return frame

    @staticmethod
    def draw_panel(frame, title, lines, origin=(12, 28), color=(30, 64, 175)):
        import cv2

        lines = [str(line) for line in lines if line is not None and str(line) != '']
        text_lines = [str(title)] + lines[:8]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in text_lines]
        panel_w = max([size[0] for size in sizes] + [120]) + 18
        panel_h = 24 + len(text_lines) * 20
        x, y = origin
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, max(0, y - 20)), (x + panel_w, y - 20 + panel_h), color, -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        for index, line in enumerate(text_lines):
            cv2.putText(frame, line, (x + 9, y + index * 20), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
        return frame

    @staticmethod
    def draw_no_output(frame, service_name, output_label):
        return ImageVisualizer.draw_panel(
            frame,
            'No output',
            [f'service: {service_name}', f'output: {output_label}'],
            color=(80, 80, 80),
        )
