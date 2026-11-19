"""Streamlit UI: mode switch, chat, telemetry cards, 7-element context bar, KV-cache hit, context budget, MCP prompt/resource buttons."""
import streamlit as st
from core import config
from core.graph import OpsGate
from core.telemetry import ELEMENTS
from core.tracing import setup_tracing

st.set_page_config(page_title="OpsGate Context Copilot", layout="wide")
COLORS = {"instructions": "#6c5ce7", "history": "#00b894", "prompt": "#fdcb6e", "tool_schemas": "#b2bec3", "tool_io": "#636e72", "rag": "#0984e3", "memory": "#e17055"}

@st.cache_resource(show_spinner="Connecting to MCP servers...")
def get_agent(): return OpsGate()
agent = get_agent()

def stacked_bar(elements: dict):
    total = max(1, sum(elements.values()))
    segs = "".join(f'<div style="width:{100*v/total:.2f}%;background:{COLORS[e]};min-width:{2 if v else 0}px" title="{e}: {v}"></div>' for e, v in elements.items() if v)
    legend = "".join(f'<span style="margin-right:14px;font-size:12px"><span style="display:inline-block;width:10px;height:10px;background:{COLORS[e]};border-radius:2px;margin-right:5px"></span>{e.replace("_", " ")} {elements[e]:,}</span>' for e in ELEMENTS)
    st.markdown(f'<div style="display:flex;height:30px;border-radius:6px;overflow:hidden;margin:6px 0">{segs}</div><div>{legend}</div>', unsafe_allow_html=True)

with st.sidebar:
    st.title("OpsGate")
    mode = st.radio("Mode", ["engineered", "naive"], format_func=lambda m: "OpsGate Context-Engineered" if m == "engineered" else "Naive Baseline")
    user_id = st.text_input("User ID", "demo")
    if st.button("End session (extract memories)"):
        st.success(f"Stored {len(agent.end_session(user_id, mode, f'{user_id}-{mode}-{st.session_state.get('thread', 0)}'))} memories")
    if st.button("Clear chat"):
        st.session_state.pop("log", None); st.session_state["thread"] = st.session_state.get("thread", 0) + 1; st.session_state.pop("last", None)
    st.divider(); st.caption("MCP resources / prompts")
    if st.button("Show SLA policy (MCP resource)"): st.info(agent.host.read_resource("incident", "service://sla-policy"))
    svc = st.selectbox("Service", ["payment-worker", "order-api", "inventory-sync"])
    if st.button("Postmortem prompt (MCP prompt)"): st.session_state["prefill"] = agent.host.get_prompt("incident", "postmortem_template", {"service": svc})
    ls = setup_tracing()
    st.divider(); st.caption("LangSmith tracing: " + (f"ON, project {ls['project']} ({'inputs/outputs hidden' if ls['redacted'] else 'FULL data sent'})" if ls["enabled"] else "off (add LANGSMITH_API_KEY to .env)"))
    st.caption("Memory loaded at session start (static system prompt)")
    try:
        sem, proc = agent.mem(user_id).bootstrap()
        st.markdown("\n".join(f"- {x}" for x in (proc + sem)) or "_nothing stored yet_")
    except Exception as e: st.caption(f"memory unavailable: {e}")

st.session_state.setdefault("log", []); st.session_state.setdefault("runs", {})
thread = f"{user_id}-{mode}-{st.session_state.get('thread', 0)}"
for role, text in st.session_state["log"]: st.chat_message(role).write(text)

q = st.chat_input("Ask about runbooks, failed orders, commits...") or st.session_state.pop("prefill", None)
if q:
    st.chat_message("user").write(q); st.session_state["log"].append(("user", q))
    with st.spinner("Thinking..."):
        r = agent.ask(q, mode, user_id, thread)
    st.chat_message("assistant").write(r["answer"]); st.session_state["log"].append(("assistant", r["answer"]))
    st.session_state["last"] = (q, mode, r["telemetry"]); st.session_state["runs"][(q, mode)] = r["telemetry"]

if "last" in st.session_state:
    q0, m0, t = st.session_state["last"]
    other = st.session_state["runs"].get((q0, "naive" if m0 == "engineered" else "engineered"))     # same prompt, other mode -> live comparison
    delta = f"{100 * (t['input_tokens'] - other['input_tokens']) / max(1, other['input_tokens']):+.0f}% vs {other['mode']} ({other['input_tokens']:,})" if other else "ask this in the other mode to compare"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Context tokens", f"{t['input_tokens']:,}", delta, delta_color="inverse" if other else "off", help=f"route: {t['route']} ({t['route_reason']}) | LLM calls: {t['llm_calls']}")
    c2.metric("Tools loaded", f"{t['tools_in_prompt']} / {t['total_tools']}")
    c3.metric("Retrieval", f"score {t['top_score']}" if t["top_score"] is not None else "not used", f"{t['chunks_kept']} chunks kept")
    c4.metric("KV-cache hit", f"{t['cache_hit_pct']}%", f"{t['cached_tokens']:,} of {t['sent_tokens']:,} sent ({t['cache_source']})",
              help="Share of prompt tokens served from the provider's KV cache. Static prefix stays identical; only the bottom changes.")
    # click a card's "Details" to open its breakdown (all local; nothing here is sent anywhere)
    with c1.popover("Details", use_container_width=True):
        tot = max(1, sum(t["elements"].values()))
        st.markdown(f"**Route:** `{t['route']}` chosen by `{t['route_reason'] or 'n/a'}`")
        st.dataframe([{"element": e.replace("_", " "), "tokens": t["elements"][e], "share %": round(100 * t["elements"][e] / tot, 1)} for e in ELEMENTS], hide_index=True)
        st.caption(f"Cleaner saved ~{t['cleaner_saved']} | summary saved ~{t['summary_saved']} | tool output truncated by cap: {t['truncated_tokens']} tokens")
    with c2.popover("Details", use_container_width=True):
        if t["tool_calls"]:
            st.dataframe([{k: c[k] for k in ("n", "tool", "args", "ms", "out_tokens", "status")} for c in t["tool_calls"]], hide_index=True)
            for c in t["tool_calls"]:
                with st.expander(f"#{c['n']} {c['tool']} output preview"): st.code(c["preview"])
        else: st.write("No tools were called in this run.")
    with c3.popover("Details", use_container_width=True):
        st.dataframe(t["chunks"], hide_index=True) if t["chunks"] else st.write("Retrieval was not used, or nothing passed the similarity threshold.")
    with c4.popover("Details", use_container_width=True):
        st.markdown(f"Static prefix (system + tools): **{t['prefix_tokens']:,}** tokens | source: `{t['cache_source']}`")
        st.dataframe(t["llm_log"], hide_index=True) if t["llm_log"] else st.write("No LLM call was needed.")
    st.caption(f"Cleaner saved ~{t['cleaner_saved']} tokens | summary saved ~{t['summary_saved']} | billed-equivalent tokens {t['effective_tokens']:,} | static prefix {t['prefix_tokens']:,} | latency {t['latency_s']}s")
    st.markdown("**Context elements (tokens)**"); stacked_bar(t["elements"])
    st.markdown(f"**Context budget** · {t['input_tokens']:,} / {config.CONTEXT_BUDGET_TOKENS:,}")
    st.progress(min(1.0, t["input_tokens"] / config.CONTEXT_BUDGET_TOKENS))
