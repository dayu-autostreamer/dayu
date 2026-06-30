import os

workers = 4

worker_class = "uvicorn.workers.UvicornWorker"

bind = f"0.0.0.0:{os.getenv('GUNICORN_PORT', 8910)}"

timeout = int(os.getenv("GUNICORN_TIMEOUT", 1200))

graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", timeout))

accesslog = '-'

errorlog = '-'
