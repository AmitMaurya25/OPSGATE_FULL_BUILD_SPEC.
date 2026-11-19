"""LangGraph "Watchman" state machine. mode = "naive" | "engineered"."""
import json, re, time
from typing import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from core import config
from core.llm import get_llm, have_llm, IGNORANCE
from core.mcp_host import MCPHost
from core.tool_registry import ToolRegistry, schema_of
from core.rag_retriever import retrieve
from core.cleaner import clean_context, cap_tokens
from core.tracing import setup_tracing, trace_config
from core.sandbox import run_programmatic_task
from core.summarizer import should_summarize, summarize, split_distal, total_tokens
from core.telemetry import Telemetry, count_tokens
from core.cache import PrefixCache, render_prompt
from core.router import route_by_rules
from core.prompts import build_system, build_user, SYSTEM_NAIVE, ROUTER_PROMPT
from memory.manager import MemoryManager

class S(TypedDict, total=False):
    question: str; mode: str; user_id: str
    messages: list; route: str; answer: str; tel: Telemetry
    loaded: list            # tools already loaded in this thread (sticky, most recent last, capped)

setup_tracing()          # no-op unless LANGSMITH_API_KEY is set
MAX_STICKY_TOOLS = 4

class OpsGate:
    def __init__(self, host: MCPHost | None = None):
        self.host = host or MCPHost().load()
        self.registry = ToolRegistry(self.host.tools)
        self.search_tool = self.registry.make_search_tool()
        self.llm = get_llm()
        self.memory: dict[str, MemoryManager] = {}
        self.cache = {"naive": PrefixCache(), "engineered": PrefixCache()}     # KV-cache hit estimate, one history per mode
        self.reference = self._load_reference()
        self.graph = self._build()

    # ---------- helpers ----------
    def mem(self, uid):
        if uid not in self.memory: self.memory[uid] = MemoryManager(uid)
        return self.memory[uid]
    def _load_reference(self) -> str:
        """Small static docs (SLA policy, service catalog) go straight into the system prompt, not RAG: they are tiny, always
        relevant and identical for every user, so they also sit inside the cached static prefix."""
        try:
            return "\n".join([self.host.read_resource("incident", "service://sla-policy"), self.host.read_resource("incident", "service://service-catalog")])
        except Exception: return ""
    def _call_llm(self, msgs, tel, tools=None):
        llm = self.llm.bind_tools(tools) if tools else self.llm
        tel.llm_calls += 1
        total, cached = self.cache.get(tel.mode, self.cache["engineered"]).observe(render_prompt(msgs, tools))
        if tel.llm_calls == 1:                                           # size of the static prefix (system prompt + tool schemas)
            tel.prefix_tokens = count_tokens(msgs[0].content if getattr(msgs[0], "type", "") == "system" else "") + \
                self.registry.schema_tokens([getattr(t, "name", "") for t in tools or [] if getattr(t, "name", "") in self.registry.entries])
        t0 = time.time(); resp = llm.invoke(msgs)
        tel.llm_log.append({"n": tel.llm_calls, "sent": total, "cached": cached, "ms": round(1000 * (time.time() - t0))})
        tel.record_call(total, cached, getattr(resp, "usage_metadata", None))
        return resp
    def _account(self, tel, history, instructions):
        for m in history: tel.add("history", m.content)
        tel.add("instructions", instructions)
    @staticmethod
    def _cap(text, tel):
        text, dropped = cap_tokens(text, config.MAX_TOOL_OUTPUT_TOKENS); tel.truncated_tokens += dropped; return text
    def _run_tool(self, name, args, engineered, tel):
        try:
            schema = self.registry.entries[name]["schema"]["parameters"]
            paged = "page_size" in json.dumps(schema)
            if engineered and paged:
                out = run_programmatic_task(self.host, name, args)
                tel.cleaner_saved += max(0, (out["raw_chars"] - len(out["text"])) // 4)
                return self._cap(out["text"], tel)
            res = self.host.call(name, args)
            raw = res if isinstance(res, str) else json.dumps(res, indent=2, default=str)
            if not engineered: return raw
            cleaned = clean_context(raw); tel.cleaner_saved += max(0, (len(raw) - len(cleaned)) // 4)
            return self._cap(cleaned, tel)
        except Exception as e:
            return f"tool error: {e}"
    def _bind(self, loaded):
        """search_tools first, then loaded tools in fixed registry order (stable KV prefix)."""
        return [self.search_tool] + [self.host.tools[n] for n in self.registry.ordered(loaded)]
    def _tool_loop(self, msgs, tel, engineered, loaded=None, question=""):
        loaded = list(loaded or [])
        if engineered and config.AUTO_PRELOAD_TOOLS and question:
            # host-side search_tools: the same ranking the LLM would get, without paying a whole LLM round trip to ask for it
            for n in self.registry.search(question, k=2): loaded = [x for x in loaded if x != n] + [n]
            loaded = loaded[-MAX_STICKY_TOOLS:]
        bound = self._bind(loaded) if engineered else list(self.host.tools.values())
        for _ in range(6):
            names = [getattr(t, "name", "") for t in bound]
            tel.elements["tool_schemas"] = max(tel.elements["tool_schemas"], self.registry.schema_tokens([n for n in names if n in self.registry.entries]) + (150 if engineered else 0))
            tel.tools_in_prompt = max(tel.tools_in_prompt, len(bound)); tel.total_tools = len(self.host.tools)
            resp = self._call_llm(msgs, tel, bound); msgs.append(resp)
            if not getattr(resp, "tool_calls", None): return resp.content, loaded
            for tc in resp.tool_calls:
                if tc["name"] == "search_tools":
                    found = self.registry.search(tc["args"].get("query", ""), k=2)
                    for n in found: loaded = [x for x in loaded if x != n] + [n]
                    loaded = loaded[-MAX_STICKY_TOOLS:]; bound = self._bind(loaded); out = json.dumps(found)
                elif engineered and tc["name"] not in loaded:
                    out = "Guardrail: tool not loaded. Call search_tools first."
                else:
                    t0 = time.time(); out = self._run_tool(tc["name"], tc["args"], engineered, tel)
                    tel.log_tool(tc["name"], tc["args"], out, 1000 * (time.time() - t0), "error" if out.startswith("tool error") else "ok")
                if tc["name"] == "search_tools" or (engineered and tc["name"] not in loaded and out.startswith("Guardrail")):
                    tel.log_tool(tc["name"], tc["args"], out, 0, "search" if tc["name"] == "search_tools" else "blocked")
                tel.add("tool_io", out); msgs.append(ToolMessage(out, tool_call_id=tc["id"], name=tc["name"]))
        return "Stopped after too many tool steps.", loaded

    # ---------- nodes ----------
    def bootstrap(self, s: S):
        msgs = s.get("messages") or []
        if msgs and msgs[0].type == "system": return {}
        if s["mode"] == "naive": return {"messages": [SystemMessage(SYSTEM_NAIVE)]}
        sem, proc = self.mem(s["user_id"]).bootstrap()
        return {"messages": [SystemMessage(build_system(sem, proc, self.reference))]}

    def router(self, s: S):
        tel = Telemetry(mode=s["mode"])
        if s["mode"] == "naive": tel.route = "naive"; tel.route_reason = "naive"; return {"route": "naive", "tel": tel}
        q = s["question"]
        route = route_by_rules(q); tel.route_reason = "rule" if route else ""   # free, instant, and keeps the first request prefix stable
        if route is None and have_llm():               # LLM classifier only for messages no rule recognises
            prompt = ROUTER_PROMPT.format(q=q); tel.add("prompt", prompt)
            label = self._call_llm([HumanMessage(prompt)], tel).content.strip().lower()
            route = label if label in ("chat", "runbook", "ops") else "chat"; tel.route_reason = "llm"
        route = route or "chat"; tel.route_reason = tel.route_reason or "default"
        tel.route = route; return {"route": route, "tel": tel}

    def chat(self, s: S):
        tel, msgs = s["tel"], list(s["messages"]) + [HumanMessage(s["question"])]
        self._account(tel, msgs[1:-1], msgs[0].content); tel.add("prompt", s["question"])
        ans = self._call_llm(msgs, tel).content
        return {"answer": ans, "messages": msgs + [AIMessage(ans)]}

    def runbook(self, s: S):
        tel, hist = s["tel"], list(s["messages"])
        docs = retrieve(s["question"], k=5)
        tel.chunks_kept = len(docs); tel.top_score = docs[0]["score"] if docs else None
        tel.chunks = [{"section": d["section"], "kind": d["kind"], "score": d["score"], "tokens": count_tokens(d["text"])} for d in docs]
        if not docs:                                              # ignorance clause, no LLM call needed
            return {"answer": IGNORANCE, "messages": hist + [HumanMessage(s["question"]), AIMessage(IGNORANCE)]}
        for d in docs:
            raw = d["text"]; d["text"] = clean_context(raw) if d["kind"] == "prose" else raw   # lossless only on prose; keep code/tables byte-exact
            tel.cleaner_saved += max(0, (len(raw) - len(d["text"])) // 4)
        epi = self.mem(s["user_id"]).episodic(s["question"])
        user = build_user(s["question"], docs, epi)
        self._account(tel, hist[1:], hist[0].content); tel.add("rag", "\n".join(d["text"] for d in docs)); tel.add("memory", "\n".join(epi)); tel.add("prompt", s["question"])
        ans = self._call_llm(hist + [HumanMessage(user)], tel).content
        return {"answer": ans, "messages": hist + [HumanMessage(s["question"]), AIMessage(ans)]}

    def ops(self, s: S):
        tel, hist = s["tel"], list(s["messages"])
        self._account(tel, hist[1:], hist[0].content); tel.add("prompt", s["question"])
        epi = self.mem(s["user_id"]).episodic(s["question"]); tel.add("memory", "\n".join(epi))
        msgs = hist + [HumanMessage(build_user(s["question"], None, epi))]
        ans, loaded = self._tool_loop(msgs, tel, engineered=True, loaded=s.get("loaded"), question=s["question"])
        return {"answer": ans, "loaded": loaded, "messages": hist + [HumanMessage(s["question"]), AIMessage(ans)]}

    def naive(self, s: S):
        tel, hist = s["tel"], list(s["messages"])
        docs = retrieve(s["question"], k=5, use_threshold=False)          # plain Top-K, no threshold
        tel.chunks_kept = len(docs); tel.top_score = docs[0]["score"] if docs else None
        tel.chunks = [{"section": d["section"], "kind": d["kind"], "score": d["score"], "tokens": count_tokens(d["text"])} for d in docs]
        self._account(tel, hist[1:], hist[0].content); tel.add("rag", json.dumps(docs, indent=2)); tel.add("prompt", s["question"])
        msgs = hist + [HumanMessage(build_user(s["question"], docs))]
        ans, _ = self._tool_loop(msgs, tel, engineered=False)
        return {"answer": ans, "messages": hist + [HumanMessage(s["question"]), AIMessage(ans)]}

    def maybe_summarize(self, s: S):
        if s["mode"] == "engineered" and should_summarize(s["messages"], config.SUMMARY_EVERY_N_TURNS or None):
            _, old, _ = split_distal(s["messages"])
            new, saved = summarize(s["messages"]); s["tel"].summary_saved = saved
            if old: self.mem(s["user_id"]).end_session(old)      # spec: also extract memories at the summary trigger, before detail is lost
            return {"messages": new}
        return {}

    def finalize(self, s: S): s["tel"].finish(); return {}

    def _build(self):
        g = StateGraph(S)
        for n in ["bootstrap", "router", "chat", "runbook", "ops", "naive", "maybe_summarize", "finalize"]: g.add_node(n, getattr(self, n))
        g.add_edge(START, "bootstrap"); g.add_edge("bootstrap", "router")
        g.add_conditional_edges("router", lambda s: s["route"], {"chat": "chat", "runbook": "runbook", "ops": "ops", "naive": "naive", "static": "chat"})
        for n in ["chat", "runbook", "ops", "naive"]: g.add_edge(n, "maybe_summarize")
        g.add_edge("maybe_summarize", "finalize"); g.add_edge("finalize", END)
        serde = JsonPlusSerializer(allowed_msgpack_modules=[("core.telemetry", "Telemetry")])   # Telemetry rides in the checkpointed state: allow it explicitly
        return g.compile(checkpointer=MemorySaver(serde=serde))

    # ---------- public API ----------
    def ask(self, question: str, mode: str = "engineered", user_id: str = "demo", thread_id: str | None = None) -> dict:
        cfg = {"configurable": {"thread_id": thread_id or f"{user_id}-{mode}"}}
        cfg.update(trace_config(mode, user_id, cfg["configurable"]["thread_id"]))
        out = self.graph.invoke({"question": question, "mode": mode, "user_id": user_id}, cfg)
        return {"answer": out["answer"], "telemetry": out["tel"].as_dict()}
    def end_session(self, user_id: str, mode: str = "engineered", thread_id: str | None = None):
        cfg = {"configurable": {"thread_id": thread_id or f"{user_id}-{mode}"}}
        st = self.graph.get_state(cfg).values
        return self.mem(user_id).end_session(st.get("messages", [])) if st else []
