import uvicorn

from core.lib.common import Context
from core.scheduler import SchedulerServer


app = SchedulerServer().app


if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=Context.get_parameter('GUNICORN_PORT', 9400, direct=False),
        workers=1,
    )
