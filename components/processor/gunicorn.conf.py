import os

workers = 1

worker_class = "uvicorn.workers.UvicornWorker"

bind = f"0.0.0.0:{os.getenv('GUNICORN_PORT', 9000)}"

# Processor inference runs in the worker process and some models have a long
# first-use warm-up.  Gunicorn's 30-second default can therefore kill an
# otherwise healthy worker while it still owns the in-memory FIFO, dropping
# both the running invocation and queued branches.  Keep the watchdog bounded,
# but comfortably above the profiled cold-start envelope.  Deployments may
# override this when a different application has a known larger bound.
timeout = int(os.getenv("GUNICORN_TIMEOUT", "300"))

accesslog = '-'

errorlog = '-'
