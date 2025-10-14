import os
import cv2
import json
import uvicorn
import argparse
import socket
import requests
import threading
import time
import asyncio
from pydantic import BaseModel

from fastapi import FastAPI, Form, BackgroundTasks
from fastapi.routing import APIRouter
from starlette.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from core.lib.common import FileOps, LOGGER, Context, NameMaintainer

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
sources = {}


class SourceRequest(BaseModel):
    root: str
    path: str
    play_mode: str


class VideoSource:
    def __init__(self, data_root, play_mode):
        self.router = APIRouter()
        self.router.add_api_route('/source', self.get_source_data, methods=['GET'])
        self.router.add_api_route('/file', self.get_source_file, methods=['GET'])

        self.data_root = data_root
        self.data_dir = os.path.join(self.data_root, 'frames')

        self.play_mode = play_mode

        self.frame_count = 0
        self.is_end = False
        self.frame_max_count = len(os.listdir(self.data_dir))

        self.file_name = None

        self.source_id = None
        self.task_id = None
        self.meta_data = None
        self.raw_meta_data = None

        self.frame_filter = None
        self.frame_process = None
        self.frame_compress = None

        self.file_suffix = 'mp4'

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

        self.source_id = data['source_id']
        self.task_id = data['task_id']
        self.meta_data = data['meta_data']
        self.raw_meta_data = data['raw_meta_data']

        frame_filter_name = data['gen_filter_name']
        frame_process_name = data['gen_process_name']
        frame_compress_name = data['gen_compress_name']

        buffer_size = self.meta_data['buffer_size']

        self.frame_filter = Context.get_algorithm('GEN_FILTER', al_name=frame_filter_name) \
            if self.frame_filter is None else self.frame_filter
        self.frame_process = Context.get_algorithm('GEN_PROCESS', al_name=frame_process_name) \
            if self.frame_process is None else self.frame_process
        self.frame_compress = Context.get_algorithm('GEN_COMPRESS', al_name=frame_compress_name) \
            if self.frame_compress is None else self.frame_compress

        frames_index = []
        frames_buffer = []
        while len(frames_buffer) < buffer_size:
            frame = self.get_one_frame()
            if self.frame_filter(self, frame):
                frames_buffer.append(frame)
                frames_index.append(self.frame_count)

        frames_buffer = [
            self.frame_process(self, frame, self.raw_meta_data['resolution'], self.meta_data['resolution'])
            for frame in frames_buffer
        ]

        self.file_name = NameMaintainer.get_task_data_file_name(self.source_id, self.task_id,
                                                                file_suffix=self.file_suffix)
        self.frame_compress(self, frames_buffer, self.file_name)

        return JSONResponse(frames_index)

    def get_source_file(self, backtask: BackgroundTasks):
        return FileResponse(path=self.file_name, filename=self.file_name, media_type='application/octet-stream',
                            background=backtask.add_task(FileOps.remove_file, self.file_name))


@app.post("/admin/add_source")
async def add_source(request: SourceRequest):
    if request.path in sources:
        return {"status": "error", "message": "Path already exists"}
    source = VideoSource(request.root, request.play_mode)
    app.include_router(source.router, prefix=f"/{request.path}")
    sources[request.path] = source
    return {"status": "success"}


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def wait_for_port(port: int, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_in_use(port):
            return True
        time.sleep(0.5)
    return False


def run_server(port: int):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    config = uvicorn.Config(app, host="0.0.0.0", port=port)
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())


def register_source(root: str, path: str, play_mode: str):
    try:
        response = requests.post(
            f"http://127.0.0.1:{server_port}/admin/add_source",
            json={"root": root, "path": path, "play_mode": play_mode}
        )
        LOGGER.info(f"{path} registered to existing server: {response.json()}")

    except Exception as e:
        LOGGER.warning(f"{path} failed to register: {str(e)}")
        LOGGER.exception(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--address', type=str, required=True)
    parser.add_argument('--play_mode', type=str, required=True)
    args = parser.parse_args()

    server_port = int(args.address.split(':')[-1].split('/')[0])
    server_path = args.address.split('/')[-1]

    if is_port_in_use(server_port):
        # server already in running
        register_source(args.root, server_path, args.play_mode)
    else:
        # first run server
        server_thread = threading.Thread(target=run_server, args=(server_port,), daemon=True)
        server_thread.start()
        if wait_for_port(server_port):
            register_source(args.root, server_path, args.play_mode)
            server_thread.join()
        else:
            LOGGER.warning(f"Failed to start server on port {server_port}")
