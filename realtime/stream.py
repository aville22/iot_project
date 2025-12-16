import asyncio
from typing import Dict


class EventStream:
    """
    Simple async event queue for IoT data.
    """
    def __init__(self, maxsize: int = 1000):
        self.queue = asyncio.Queue(maxsize=maxsize)

    async def publish(self, event: Dict) -> None:
        await self.queue.put(event)

    async def consume(self) -> Dict:
        return await self.queue.get()
