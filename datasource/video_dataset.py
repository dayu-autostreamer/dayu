import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from core.lib.common import LOGGER

MANIFEST_FILE_NAME = 'manifest.json'
DEFAULT_VIDEO_ROOT = '../data'
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.ts', '.webm', '.flv'}


def is_video_file(path):
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def probe_video_frame_count(video_path):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f'Failed to open video "{video_path}".')

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if frame_count > 0:
        return frame_count

    command = [
        'ffprobe',
        '-v', 'error',
        '-count_frames',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames',
        '-of', 'default=nokey=1:noprint_wrappers=1',
        str(video_path),
    ]
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0
    return int(output) if output.isdigit() else 0


@dataclass(frozen=True)
class VideoClip:
    name: str
    path: str
    frame_count: int
    start_frame_index: int

    @property
    def end_frame_index(self):
        return self.start_frame_index + self.frame_count - 1


class VideoDataset:
    def __init__(self, manifest_dir):
        self.manifest_dir = Path(manifest_dir).resolve()
        self.manifest_path = self.manifest_dir / MANIFEST_FILE_NAME
        if not self.manifest_path.is_file():
            raise ValueError(f'Missing manifest file "{self.manifest_path}".')

        with open(self.manifest_path, 'r', encoding='utf-8') as file:
            manifest = json.load(file)

        version = manifest.get('version', 1)
        if version != 1:
            raise ValueError(f'Unsupported manifest version "{version}" in "{self.manifest_path}", expected version 1.')

        manifest_type = manifest.get('type', 'video_sequence')
        if manifest_type != 'video_sequence':
            raise ValueError(f'Unsupported manifest type "{manifest_type}" in "{self.manifest_path}", '
                             f'expected type "video_sequence".')

        self.video_root = (self.manifest_dir / manifest.get('video_root', DEFAULT_VIDEO_ROOT)).resolve()
        if not self.video_root.exists():
            raise ValueError(f'Video root "{self.video_root}" does not exist for manifest "{self.manifest_path}".')

        sequence = manifest.get('sequence')
        if not isinstance(sequence, list) or not sequence:
            raise ValueError(f'Manifest "{self.manifest_path}" must provide a non-empty "sequence" list.')

        self.clips = self._build_clips(sequence)
        self.total_frames = sum(clip.frame_count for clip in self.clips)
        LOGGER.info(
            f'Loaded video dataset from "{self.manifest_path}": '
            f'{len(self.clips)} clips, {self.total_frames} decodable frames.'
        )

    def _build_clips(self, sequence):
        clips = []
        next_frame_index = None

        for clip_index, entry in enumerate(sequence):
            if not isinstance(entry, dict):
                raise ValueError(f'Invalid clip entry #{clip_index} in "{self.manifest_path}": {entry}')
            relative_path = entry.get('path')
            if not relative_path:
                raise ValueError(f'Clip entry #{clip_index} in "{self.manifest_path}" is missing "path".')

            clip_path = (self.video_root / relative_path).resolve()
            if not clip_path.is_file():
                raise ValueError(f'Clip "{relative_path}" from "{self.manifest_path}" does not exist.')
            if not is_video_file(clip_path):
                raise ValueError(f'Clip "{relative_path}" from "{self.manifest_path}" is not a supported video.')

            frame_count = int(entry.get('frame_count') or probe_video_frame_count(clip_path))
            if frame_count <= 0:
                raise ValueError(f'Failed to resolve frame count for clip "{clip_path}".')

            if 'start_frame_index' in entry:
                start_frame_index = int(entry['start_frame_index'])
            else:
                start_frame_index = 0 if next_frame_index is None else next_frame_index
            if start_frame_index < 0:
                raise ValueError(f'Clip "{relative_path}" in "{self.manifest_path}" has a negative start_frame_index.')

            clip = VideoClip(
                name=entry.get('name', clip_path.stem),
                path=str(clip_path),
                frame_count=frame_count,
                start_frame_index=start_frame_index,
            )
            clips.append(clip)
            next_frame_index = clip.start_frame_index + clip.frame_count

        return clips

    def get_clip_paths(self):
        return [clip.path for clip in self.clips]


class VideoDatasetPlayer:
    def __init__(self, manifest_dir, play_mode):
        self.dataset = VideoDataset(manifest_dir)
        self.play_mode = play_mode
        self.current_clip_index = 0
        self.current_frame_offset = 0
        self.capture = None
        self.is_end = False

    def read_frame(self):
        if self.is_end:
            return None, None

        while True:
            self._ensure_capture()
            ret, frame = self.capture.read()
            if ret:
                clip = self.dataset.clips[self.current_clip_index]
                frame_index = clip.start_frame_index + self.current_frame_offset
                self.current_frame_offset += 1
                if self.current_frame_offset >= clip.frame_count:
                    self._advance_after_clip_end()
                return frame, frame_index

            if not self._advance_to_next_clip():
                return None, None

    def _ensure_capture(self):
        if self.capture and self.capture.isOpened():
            return

        clip = self.dataset.clips[self.current_clip_index]
        self.capture = cv2.VideoCapture(clip.path)
        if not self.capture.isOpened():
            raise ValueError(f'Failed to open video "{clip.path}".')
        if self.current_frame_offset:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_offset)

    def _advance_after_clip_end(self):
        reached_last_clip = self.current_clip_index >= len(self.dataset.clips) - 1
        self.close()

        if reached_last_clip:
            if self.play_mode == 'non-cycle':
                self.is_end = True
                LOGGER.info('A video play cycle ends. Video play ends in non-cycle mode.')
            else:
                LOGGER.info('A video play cycle ends. Replay video in cycle mode.')
            self.current_clip_index = 0
            self.current_frame_offset = 0
            return

        self.current_clip_index += 1
        self.current_frame_offset = 0

    def _advance_to_next_clip(self):
        reached_last_clip = self.current_clip_index >= len(self.dataset.clips) - 1
        self.close()

        if reached_last_clip:
            if self.play_mode == 'non-cycle':
                self.is_end = True
                return False
            self.current_clip_index = 0
        else:
            self.current_clip_index += 1

        self.current_frame_offset = 0
        return not self.is_end

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
