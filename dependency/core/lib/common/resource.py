import asyncio
from typing import Dict, Optional


class ResourceLockManager:
    """Resource Lock Management Tool Class"""

    def __init__(self):
        # Resource lock storage structure：resource_name -> device_name
        self._locks: Dict[str, Optional[str]] = {}

        # Lazily bind the asyncio lock to the active event loop.
        self._global_lock: Optional[asyncio.Lock] = None
        self._global_lock_loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_global_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._global_lock is None or self._global_lock_loop is not loop:
            self._global_lock = asyncio.Lock()
            self._global_lock_loop = loop
        return self._global_lock

    async def acquire_lock(self, resource: str, user: str) -> Optional[str]:
        """
        Acquire resource lock:
        Return the username currently occupying resource
        """
        async with self._get_global_lock():
            current_holder = self._locks.get(resource)

            if current_holder is None:
                self._locks[resource] = user
                return user

            return current_holder

    async def release_lock(self, resource: str, user: str) -> bool:
        """Release resource lock (only release when the current holder is the requesting device)"""
        async with self._get_global_lock():
            if resource in self._locks and self._locks[resource] == user:
                self._locks[resource] = None
                return True
            return False

    async def get_current_holder(self, resource: str) -> Optional[str]:
        """Query the current resource holder"""
        async with self._get_global_lock():
            return self._locks.get(resource)
