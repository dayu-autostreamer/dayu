import os
import cv2
import json
import uvicorn
import argparse

from fastapi import FastAPI, Form, BackgroundTasks
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from core.lib.common import FileOps, LOGGER, Context


class VideoSource:
    def __init__(self, data_root, path, play_mode):
        self.app = FastAPI(routes=[
            APIRoute(f'/{path}/source',
                     self.get_source_data,
                     response_class=JSONResponse,
                     methods=['GET']
                     ),
            APIRoute(f'/{path}/file',
                     self.get_source_file,
                     response_class=FileResponse,
                     methods=['GET']
                     )
        ], log_level='trace', timeout=6000)

        self.app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"],
        )

        self.data_root = data_root
        self.data_dir = os.path.join(self.data_root, 'frames')

        self.play_mode = play_mode

        self.frame_count = 0
        self.is_end = False
        self.frame_max_count = len(os.listdir(self.data_dir))

        self.file_name = None

        self.meta_data = None
        self.raw_meta_data = None

        self.frame_filter = None
        self.frame_process = None
        self.frame_compress = None

    def get_one_frame(self):
        frame = cv2.imread(os.path.join(self.data_dir, f'{self.frame_count}.jpg'))
        self.frame_count += 1
        if self.frame_count >= self.frame_max_count:
            if self.play_mode == 'non-cycle':
                self.is_end = True
                LOGGER.info('A video play cycle ends. Video play ends in non-cycle mode.')
            else:
                LOGGER.info('A video play cycle ends. Replay video in cycle mode.')
        self.frame_count %= self.frame_max_count
        return frame

    def get_source_data(self, data: str = Form(...)):

        if self.is_end:
            return []

        data = json.loads(data)

        source_id = data['source_id']
        task_id = data['task_id']
        self.meta_data = data['meta_data']
        self.raw_meta_data = data['raw_meta_data']

        frame_filter_name = data['gen_filter_name']
        frame_process_name = data['gen_process_name']
        frame_compress_name = data['gen_compress_name']

        buffer_size = self.meta_data['buffer_size']

        self.frame_filter = Context.get_algorithm('GEN_FILTER', al_name=frame_filter_name)
        self.frame_process = Context.get_algorithm('GEN_PROCESS', al_name=frame_process_name)
        self.frame_compress = Context.get_algorithm('GEN_COMPRESS', al_name=frame_compress_name)

        frames_index = []
        frames_buffer = []
        while len(frames_buffer) < buffer_size:
            frame = self.get_one_frame()
            if self.frame_filter(self, frame):
                frames_buffer.append(frame)
                frames_index.append(self.frame_count)

        frames_buffer = [self.frame_process(self, frame) for frame in frames_buffer]
        self.file_name = self.frame_compress(self, frames_buffer, source_id, task_id)

        return frames_index

    def get_source_file(self, backtask: BackgroundTasks):
        return FileResponse(path=self.file_name, filename=self.file_name, media_type='text/plain',
                            background=backtask.add_task(FileOps.remove_file, self.file_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--address', type=str, required=True)
    parser.add_argument('--play_mode', type=str, required=True)
    args = parser.parse_args()

    server_port = int(args.address.split(':')[-1].split('/')[0])
    server_path = args.address.split(':')[-1].split('/')[-1]
    video = VideoSource(args.root, server_path, args.play_mode)
    uvicorn.run(video.app, host='0.0.0.0', port=server_port)
