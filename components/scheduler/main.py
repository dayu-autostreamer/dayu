import uvicorn

from core.lib.common import Context
from core.scheduler import SchedulerServer


def create_app():
    """Build the Scheduler HTTP application without import-time side effects."""

    return SchedulerServer().app


def main():
    uvicorn.run(
        create_app(),
        host='0.0.0.0',
        port=Context.get_parameter('GUNICORN_PORT', 9400, direct=False),
        workers=1,
    )


if __name__ == '__main__':
    main()
