import os

workers = 1

worker_class = "uvicorn.workers.UvicornWorker"

bind = f"0.0.0.0:{os.getenv('GUNICORN_PORT', 9200)}"

accesslog = '-'

errorlog = '-'
