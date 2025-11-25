import asyncio
import threading
from typing import Any


def run_async_task(coro) -> Any:
    """
    Execute an async coroutine from synchronous code, even if an event loop is running.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        result_holder = {}
        error_holder = {}

        def runner():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result_holder["result"] = new_loop.run_until_complete(coro)
            except Exception as exc:  # pragma: no cover - pass through for sync callers
                error_holder["error"] = exc
            finally:
                new_loop.close()

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()

        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder.get("result")

    new_loop = loop or asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(new_loop)
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()
        asyncio.set_event_loop(None)


__all__ = ["run_async_task"]

