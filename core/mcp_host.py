"""MCP Host: connects to all servers (stdio) via langchain-mcp-adapters, discovers tools, calls tools,
reads resources and fetches prompts."""
import asyncio, atexit, json, sys, threading
from langchain_mcp_adapters.client import MultiServerMCPClient
from core import config

SERVER_FILES = {"git": "git_ops_server.py", "db": "db_logs_server.py", "incident": "incident_server.py", "rag": "runbook_rag_server.py"}

def server_config():
    return {name: {"command": sys.executable, "args": [str(config.ROOT / "mcp_servers" / f)], "transport": "stdio"} for name, f in SERVER_FILES.items()}

def _norm(res):
    """Normalise a tool result to a Python object (dict/list/str)."""
    if isinstance(res, tuple): res = res[0]
    if isinstance(res, list) and res and isinstance(res[0], dict) and "text" in res[0]:
        texts = [b["text"] for b in res if "text" in b]
        if len(texts) > 1:                                   # list-returning tools arrive as one block per item
            try: return [json.loads(t) for t in texts]
            except Exception: res = "\n".join(texts)
        else: res = texts[0]
    if isinstance(res, str):
        try: return json.loads(res)
        except Exception: return res
    return res

class MCPHost:
    """Host side of MCP.

    persistent=True (default): ONE background event loop keeps every server process and session alive, so a tool call is a
    JSON-RPC message instead of a new subprocess start (the old per-call spawn re-imported Python and re-opened the DB each time,
    and paged tools spawned once per page). If startup fails it falls back to the stateless mode automatically.
    """
    def __init__(self, persistent: bool | None = None):
        self.client = MultiServerMCPClient(server_config())
        self.tools: dict = {}
        self.persistent = bool(config.PERSISTENT_MCP) if persistent is None else persistent
        self._loop = self._thread = self._stop = None
        self._sessions: dict = {}

    # ---- persistent mode ----
    async def _serve(self, ready: asyncio.Future):
        """Runs as ONE task so every async context is opened and closed from the same task (anyio requirement)."""
        from contextlib import AsyncExitStack
        from langchain_mcp_adapters.tools import load_mcp_tools
        try:
            async with AsyncExitStack() as stack:
                tools = {}
                for name in SERVER_FILES:
                    sess = await stack.enter_async_context(self.client.session(name))
                    self._sessions[name] = sess
                    for t in await load_mcp_tools(sess): tools[t.name] = t
                self.tools = tools; self._stop = asyncio.Event(); ready.set_result(True)
                await self._stop.wait()
        except Exception as e:
            if not ready.done(): ready.set_exception(e)
            else: raise

    def _start(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="mcp-host"); self._thread.start()
        ready = asyncio.run_coroutine_threadsafe(self._make_ready(), self._loop).result()
        self._task = asyncio.run_coroutine_threadsafe(self._serve_task(ready), self._loop)
        ready_result = asyncio.run_coroutine_threadsafe(self._await(ready), self._loop).result(timeout=90)
        atexit.register(self.close)
        return ready_result
    async def _make_ready(self): return asyncio.get_running_loop().create_future()
    async def _serve_task(self, ready): await self._serve(ready)
    @staticmethod
    async def _await(fut): return await fut

    def _run(self, coro, timeout: float = 120):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def close(self):
        if self._loop and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
            try: self._task.result(timeout=10)
            except Exception: pass
            self._stop = None

    # ---- public API ----
    def load(self):
        if self.persistent:
            try: self._start(); return self
            except Exception as e:
                print(f"[mcp] persistent sessions unavailable ({e!r}); falling back to per-call processes")
                self.persistent = False
        self.tools = {t.name: t for t in asyncio.run(self.client.get_tools())}
        return self
    def call(self, name: str, args: dict | None = None):
        coro = self.tools[name].ainvoke(args or {})
        return _norm(self._run(coro) if self.persistent else asyncio.run(coro))
    def read_resource(self, server: str, uri: str) -> str:
        async def go(s):
            r = await s.read_resource(uri); return "\n".join(c.text for c in r.contents if hasattr(c, "text"))
        return self._with_session(server, go)
    def get_prompt(self, server: str, name: str, arguments: dict) -> str:
        async def go(s):
            r = await s.get_prompt(name, arguments); return "\n".join(m.content.text for m in r.messages if hasattr(m.content, "text"))
        return self._with_session(server, go)
    def _with_session(self, server, fn):
        if self.persistent: return self._run(fn(self._sessions[server]))
        async def one():
            async with self.client.session(server) as s: return await fn(s)
        return asyncio.run(one())
