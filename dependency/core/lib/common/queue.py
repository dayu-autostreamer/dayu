import queue
from typing import List


class Queue:
    def __init__(self, maxsize=0):
        self.__queue = queue.Queue(maxsize=maxsize)

    def put(self, item: object) -> None:
        if self.__queue.full():
            self.__queue.get()
        self.__queue.put(item)

    def get(self) -> object:
        return self.__queue.get()

    def put_all(self, items: List[object]) -> None:
        for item in items:
            self.__queue.put(item)

    def get_all(self) -> List[object]:
        items = []
        while not self.__queue.empty():
            items.append(self.__queue.get())
        return items

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


