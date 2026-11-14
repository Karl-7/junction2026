"""Run a scripted policy against a live MCP server, through exactly the tools an agent gets.

    python -m policies.mcp_env --url http://localhost:8000/mcp --policy reference
"""
from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


class McpEnv:
    """Synchronous `call(tool, **args)` facade over an async MCP client session (runs in a thread)."""

    def __init__(self, url: str):
        self.url = url
        self._req: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=lambda: asyncio.run(self._loop()), daemon=True)
        self._thread.start()
        self._ready.wait(30)

    async def _loop(self):
        async with streamablehttp_client(self.url) as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                self._ready.set()
                while True:
                    item = await asyncio.to_thread(self._req.get)
                    if item is None:
                        return
                    tool, args, out = item
                    res = await s.call_tool(tool, args)
                    if res.structuredContent is not None:
                        data = res.structuredContent.get("result", res.structuredContent)
                    else:
                        data = json.loads(res.content[0].text)
                    out.put(data)

    def call(self, tool: str, **args) -> dict:
        out: queue.Queue = queue.Queue()
        self._req.put((tool, args, out))
        return out.get(timeout=60)

    def close(self):
        self._req.put(None)
        self._thread.join(5)


def main():
    from policies import baselines, reference
    pol = {"reference": reference.run, "one_step": baselines.one_step,
           "submit_ore": baselines.submit_ore, "do_nothing": baselines.do_nothing}
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/mcp")
    ap.add_argument("--policy", default="reference", choices=sorted(pol))
    a = ap.parse_args()
    env = McpEnv(a.url)
    try:
        print(json.dumps(pol[a.policy](env)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
