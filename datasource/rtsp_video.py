import argparse
import os
import signal
import subprocess
import sys
import time

from core.lib.common import LOGGER
from video_dataset import VideoDataset


class RtspSource:
    def __init__(self, data_root, rtsp_address, play_mode):
        self.dataset = VideoDataset(data_root)
        self.rtsp_address = rtsp_address
        self.play_mode = play_mode

        self.current_process = None
        self.started_mediamtx = False
        self.mediamtx_process = None
        self.running = True

    def ensure_mediamtx(self):
        if subprocess.run(['pgrep', '-x', 'mediamtx'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            LOGGER.info('mediamtx server already running.')
            return

        rtsp_root = os.getenv('RTSP_PATH', '/rtsp_server')
        mediamtx_bin = os.path.join(rtsp_root, 'mediamtx')
        mediamtx_config = os.path.join(rtsp_root, 'mediamtx.yml')

        LOGGER.info('Starting mediamtx server..')
        self.mediamtx_process = subprocess.Popen(
            [mediamtx_bin, mediamtx_config],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.started_mediamtx = True
        time.sleep(4)

    def cleanup(self):
        self.running = False

        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.current_process.kill()
        self.current_process = None

        if self.started_mediamtx and self.mediamtx_process and self.mediamtx_process.poll() is None:
            LOGGER.info(f'Stopping mediamtx (pid={self.mediamtx_process.pid})..')
            self.mediamtx_process.terminate()
            try:
                self.mediamtx_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.mediamtx_process.kill()
        self.mediamtx_process = None

    def stream_once(self):
        for clip_path in self.dataset.get_clip_paths():
            if not self.running:
                return

            LOGGER.info(f'Streaming {clip_path} to {self.rtsp_address}')
            command = [
                'ffmpeg',
                '-re',
                '-nostdin',
                '-i', clip_path,
                '-c', 'copy',
                '-f', 'rtsp',
                '-rtsp_transport', 'tcp',
                self.rtsp_address,
            ]
            self.current_process = subprocess.Popen(command)
            return_code = self.current_process.wait()
            self.current_process = None

            if return_code != 0 and self.running:
                LOGGER.warning(f'ffmpeg exited with code {return_code} while streaming "{clip_path}".')

    def run(self):
        self.ensure_mediamtx()

        while self.running:
            self.stream_once()
            if self.play_mode == 'non-cycle':
                LOGGER.info('Stream video in non-cycle mode and end.')
                break


rtsp_source = None


def handle_signal(signum, frame):
    if rtsp_source is not None:
        rtsp_source.cleanup()
    sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--address', type=str, required=True)
    parser.add_argument('--play_mode', type=str, required=True)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    rtsp_source = RtspSource(args.root, args.address, args.play_mode)
    try:
        rtsp_source.run()
    finally:
        rtsp_source.cleanup()
