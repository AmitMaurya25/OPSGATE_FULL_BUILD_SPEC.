"""Prompt layout (KV-cache friendly): STATIC TOP (system + memory facts/rules + tool schemas), DYNAMIC BOTTOM (summary, recent msgs, episodic, docs at END, question).
Never put timestamps/random ids in the static part. All dynamic blocks use XML delimiters."""
from core.llm import IGNORANCE

SYSTEM_BASE = f"""You are OpsGate, an on-call assistant for software engineers (audience: on-call engineers at a payments company).
Task: answer runbook and policy questions from the provided documents, and answer live-ops questions by using tools.
Constraints:
- SLA policy and service catalog questions: answer from the <reference> block in this prompt.
- Runbook/policy answers must use ONLY the text inside <documents>. Cite the section name. If the documents do not contain the answer, reply exactly: "{IGNORANCE}"
- For greetings and small talk reply briefly without tools.
- Never invent tool output, error codes or numbers.
Style: concise; use compact tables for multi-row data; show code exactly as given in the documents."""

SYSTEM_NAIVE = "You are an assistant. Use the tools and documents provided."

def build_system(semantic: list[str] | None = None, procedural: list[str] | None = None, reference: str = "") -> str:
    """New System Instruction = Old System Instruction + Rules (procedural) + Facts (semantic).
    KV-cache order: the parts shared by EVERY user come first (base, reference); per-user parts come last, each list sorted so the
    same memories always render to the same bytes."""
    s = SYSTEM_BASE
    if reference: s += "\n\n<reference>\n" + reference.strip() + "\n</reference>"
    procedural, semantic = sorted(set(procedural or [])), sorted(set(semantic or []))
    if procedural: s += "\n\nRULES (user preferences):\n" + "\n".join(f"- {p}" for p in procedural)
    if semantic: s += "\n\nFACTS (about the user):\n" + "\n".join(f"- {f}" for f in semantic)
    return s

def build_user(question: str, docs: list[dict] | None = None, episodic: list[str] | None = None) -> str:
    parts = []
    if episodic: parts.append("<episodic_memory>\n" + "\n".join(f"- {e}" for e in episodic) + "\n</episodic_memory>")
    if docs is not None:                                         # documents go at the END, just before the question
        body = "\n".join(f'<document id="{i}" section="{d.get("section","")}">\n{d["text"]}\n</document>' for i, d in enumerate(docs, 1))
        parts.append(f"<documents>\n{body}\n</documents>")
    parts.append(f"<question>\n{question}\n</question>")
    return "\n\n".join(parts)

ROUTER_PROMPT = """Classify the user message into exactly one label:
runbook = knowledge/policy/how-to question answerable from runbooks (error codes, fixes, escalation, holidays/policies)
ops = needs live data or an action (database, git, alerts, on-call, rollback)
chat = greeting or general conversation
Reply with only the label.

<message>
{q}
</message>"""
