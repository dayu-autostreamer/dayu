import queue
from typing import List


class Queue:
    def __init__(self, maxsize=0):
        self.__queue = queue.Queue(maxsize=maxsize)

    def put(self, item: object) -> None:
        if self.__queue.full():
            try:
                self.__queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.__queue.put_nowait(item)
        except queue.Full:
            pass

    def get(self) -> object:
        return self.__queue.get_nowait()

    def put_all(self, items: List[object]) -> None:
        for item in items:
            self.put(item)

    def get_all(self) -> List[object]:
        out_items = []
        while True:
            try:
                out_items.append(self.get())
            except queue.Empty:
                break
        return out_items

    def get_all_without_drop(self) -> List[object]:
        # This does not affect the queue state because it doesn't consume the items
        with self.__queue.mutex:  # Lock to safely access the internal queue
            items = list(self.__queue.queue)
        return items

    def empty(self) -> bool:
        return self.__queue.empty()

    def full(self) -> bool:
        return self.__queue.full()

    def size(self) -> int:
        return self.__queue.qsize()

    def clear(self) -> None:
        self.get_all()


