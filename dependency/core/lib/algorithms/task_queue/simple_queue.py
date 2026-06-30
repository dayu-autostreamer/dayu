import abc
import threading
from queue import Queue

from core.lib.common import ClassFactory, ClassType
from core.lib.content import Task
from .base_queue import BaseQueue

__all__ = ('SimpleQueue',)


@ClassFactory.register(ClassType.PRO_QUEUE, alias='simple')
class SimpleQueue(BaseQueue, abc.ABC):
    def __init__(self):
        self._queue = Queue()
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            if self._queue.empty():
                return None
            return self._queue.get()

    def put(self, task: Task) -> None:
        with self.lock:
            self._queue.put(task)

    def size(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def drain(self, max_count=None):
        tasks = []
        with self.lock:
            while not self._queue.empty():
                if max_count is not None and len(tasks) >= max_count:
                    break
                tasks.append(self._queue.get())
        return tasks

    def get_all_without_drop(self):
        with self.lock:
            with self._queue.mutex:
                return list(self._queue.queue)

    def clear(self):
        self.drain()
