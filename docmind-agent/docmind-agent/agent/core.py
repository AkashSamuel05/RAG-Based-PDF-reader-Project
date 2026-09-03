"""
DocMind Agent Core  —  agent/core.py
=====================================
This module implements the three security layers described in the brief:

  1. TOOL REGISTRY  — every tool the agent can call is declared here with
     an explicit description, input schema, and permission scope. The agent
     cannot call anything not in this registry.

  2. PERMISSION GATE  — each tool call is checked against the session's
     granted_permissions set before execution. If the tool's required scope
     is not granted, the call is blocked and logged.

  3. PROMPT INJECTION DETECTOR  — document text passes through a sanitizer
     before reaching the agent. Injected instructions like "ignore previous
     instructions" or "send an email to" are flagged and stripped, and a
     warning is appended to the audit log.

  4. AUDIT LOG  — every agent decision, tool call, permission check, and
     injection flag is written to an in-memory log (also flushed to
     logs/audit.jsonl) with a timestamp and session ID so a human reviewer
     can inspect exactly what the agent did and why.

  5. HUMAN APPROVAL QUEUE  — any tool tagged requires_approval=True is held
     in a pending queue. The action only executes after a human explicitly
     approves it via POST /agent/approve/{call_id}.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

AUDIT_FILE = Path("logs/audit.jsonl")
AUDIT_FILE.parent.mkdir(exist_ok=True)

# ── AUDIT LOG ─────────────────────────────────────────────────────────────────

_audit: list[dict] = []          # in-memory ring buffer
_pending: dict[str, dict] = {}   # call_id → pending approval entry


def _write_audit(entry: dict):
    _audit.append(entry)
    # Keep last 500 entries in memory
    if len(_audit) > 500:
        _audit.pop(0)
    # Persist to JSONL
    with AUDIT_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def audit(event: str, session_id: str, detail: dict):
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "session_id": session_id,
        **detail,
    }
    _write_audit(entry)
    log.info("[AUDIT] %s | session=%s | %s", event, session_id, detail)
    return entry


def get_audit_log(session_id: Optional[str] = None, limit: int = 100) -> list[dict]:
    entries = _audit if session_id is None else [e for e in _audit if e.get("session_id") == session_id]
    return entries[-limit:]


# ── TOOL DEFINITION ───────────────────────────────────────────────────────────

class Tool:
    """
    A declared, scoped capability the agent may call.

    Attributes
    ----------
    name            : machine-readable identifier
    description     : shown to the agent so it knows what the tool does
    required_scope  : permission that must be granted to use this tool
    requires_approval: if True, execution is held for human approval
    fn              : the actual Python function to execute
    """

    def __init__(
        self,
        name: str,
        description: str,
        required_scope: str,
        fn: Callable,
        requires_approval: bool = False,
        input_schema: Optional[dict] = None,
    ):
        self.name = name
        self.description = description
        self.required_scope = required_scope
        self.requires_approval = requires_approval
        self.fn = fn
        self.input_schema = input_schema or {}


# ── TOOL REGISTRY ─────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Central registry of all tools the agent is allowed to call.
    Nothing outside this registry can be called by the agent.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool
        log.info("Tool registered: %s (scope=%s, approval=%s)",
                 tool.name, tool.required_scope, tool.requires_approval)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "required_scope": t.required_scope,
                "requires_approval": t.requires_approval,
            }
            for t in self._tools.values()
        ]


# ── PERMISSION GATE ───────────────────────────────────────────────────────────

class PermissionGate:
    """
    Every tool call passes through here.
    If the session does not have the required scope, the call is blocked.
    This is the core of 'defined scope' from the brief.
    """

    def check(self, tool: Tool, session_permissions: set[str], session_id: str) -> bool:
        granted = tool.required_scope in session_permissions
        audit(
            "permission_check",
            session_id,
            {
                "tool": tool.name,
                "required_scope": tool.required_scope,
                "granted_permissions": list(session_permissions),
                "result": "ALLOW" if granted else "DENY",
            },
        )
        return granted


# ── PROMPT INJECTION DETECTOR ─────────────────────────────────────────────────

# Patterns that indicate an attempt to hijack the agent via document content.
# These are conservative — they flag, not crash, so legitimate content is
# preserved and a human can review the warning in the audit log.
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"you\s+are\s+now\s+a\s+",
    r"new\s+instructions?:",
    r"system\s*:\s*you",
    r"act\s+as\s+(if\s+you\s+are\s+)?a\s+",
    r"send\s+(an?\s+)?email\s+to",
    r"delete\s+(the\s+)?(file|database|record)",
    r"drop\s+table",
    r"rm\s+-rf",
    r"os\.system\(",
    r"<\s*script",
    r"eval\s*\(",
    r"exec\s*\(",
    r"import\s+os",
    r"forget\s+(everything|all)\s+",
    r"your\s+real\s+instructions\s+are",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def sanitize_document(text: str, session_id: str) -> tuple[str, list[str]]:
    """
    Scan document text for injection patterns.
    Returns (sanitized_text, list_of_warnings).
    Flagged lines are replaced with [REDACTED — injection pattern detected].
    """
    warnings = []
    clean_lines = []
    for i, line in enumerate(text.splitlines(), 1):
        m = _INJECTION_RE.search(line)
        if m:
            warn = f"Line {i}: injection pattern '{m.group()[:40]}' detected and redacted"
            warnings.append(warn)
            clean_lines.append(f"[REDACTED — injection pattern detected on line {i}]")
        else:
            clean_lines.append(line)

    if warnings:
        audit(
            "injection_detected",
            session_id,
            {"warning_count": len(warnings), "warnings": warnings},
        )

    return "\n".join(clean_lines), warnings


# ── CHUNK SETTINGS ────────────────────────────────────────────────────────────

VALID_CHUNK_SIZES = [250, 500, 750, 1000, 1500, 2000]
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[dict]:
    """
    Split text into overlapping chunks for RAG retrieval.
    Returns list of {chunk_id, text, start_char, end_char, word_count}.
    """
    chunk_size = max(100, min(chunk_size, 3000))
    overlap = max(0, min(overlap, chunk_size // 2))
    words = text.split()
    chunks = []
    i = 0
    chunk_id = 0
    while i < len(words):
        chunk_words = words[i: i + chunk_size]
        chunk_text_str = " ".join(chunk_words)
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text_str,
            "word_start": i,
            "word_end": i + len(chunk_words),
            "word_count": len(chunk_words),
        })
        chunk_id += 1
        i += chunk_size - overlap
    return chunks


def retrieve_chunks(chunks: list[dict], question: str, top_k: int = 5) -> list[dict]:
    """
    Simple keyword-based retrieval — no embeddings needed, works fully offline.
    Scores each chunk by how many question words appear in it.
    Returns top_k chunks sorted by score descending.
    """
    q_words = set(re.findall(r'\w+', question.lower()))
    # Remove common stop words
    stop = {"the","a","an","is","are","was","were","be","been","being",
            "have","has","had","do","does","did","will","would","could",
            "should","may","might","shall","can","need","dare","ought",
            "used","to","of","in","on","at","by","for","with","about",
            "what","where","when","who","how","why","which","that","this",
            "and","or","but","if","then","so","as","it","its","i","my",
            "me","we","us","you","your","he","she","they","them","their"}
    q_words -= stop

    scored = []
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk["text"].lower()))
        if not q_words:
            score = 0
        else:
            matches = q_words & chunk_words
            score = len(matches) / len(q_words)
            # Boost score for exact phrase matches
            for word in q_words:
                if word in chunk["text"].lower():
                    score += 0.1
        scored.append({**chunk, "score": round(score, 3)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:top_k]
    # Always include at least one chunk even if score is 0
    return top if top else chunks[:1]


# ── AGENT EXECUTOR ────────────────────────────────────────────────────────────

class AgentExecutor:
    """
    Runs the agent loop:
      1. Given a question and document chunks, ask the model which tool to use.
      2. Check permissions via PermissionGate.
      3. If requires_approval → hold in pending queue.
      4. Otherwise execute the tool and return the result.

    The model's "tool selection" here is implemented as a routing decision
    based on question intent — this keeps the demo dependency-free (no
    LangChain, no OpenAI function calling) while illustrating the exact
    same security architecture.
    """

    def __init__(self, registry: ToolRegistry, gate: PermissionGate):
        self.registry = registry
        self.gate = gate

    def decide_tool(self, question: str, available_tools: list[str]) -> str:
        """
        Route the question to the appropriate tool based on intent keywords.
        In a full system this would be an LLM function-calling decision.
        """
        q = question.lower()
        if any(w in q for w in ["summarize", "summarise", "summary", "overview", "brief", "gist", "tldr"]):
            return "summarize_document"
        if any(w in q for w in ["quiz", "question", "test me", "examine"]):
            return "generate_quiz"
        if any(w in q for w in ["key term", "keyword", "key word", "important word", "extract term"]):
            return "extract_keyterms"
        if any(w in q for w in ["chunk", "split", "segment", "passage"]):
            return "retrieve_chunks"
        return "answer_question"  # default

    def run(
        self,
        session_id: str,
        question: str,
        session_permissions: set[str],
        tool_input: dict,
    ) -> dict:
        """
        Execute one agent turn. Returns result dict with keys:
        status, tool_used, result, call_id, requires_approval.
        """
        call_id = uuid.uuid4().hex[:12]

        # 1. Decide which tool to use
        available = list(self.registry._tools.keys())
        tool_name = self.decide_tool(question, available)
        tool = self.registry.get(tool_name)

        audit("agent_decision", session_id, {
            "call_id": call_id,
            "question_preview": question[:120],
            "selected_tool": tool_name,
        })

        if not tool:
            audit("tool_not_found", session_id, {"call_id": call_id, "tool": tool_name})
            return {"status": "error", "call_id": call_id, "message": f"Tool '{tool_name}' not in registry."}

        # 2. Permission check
        if not self.gate.check(tool, session_permissions, session_id):
            return {
                "status": "denied",
                "call_id": call_id,
                "tool_used": tool_name,
                "message": f"Permission denied. Tool '{tool_name}' requires scope '{tool.required_scope}'.",
            }

        # 3. Human approval gate
        if tool.requires_approval:
            _pending[call_id] = {
                "call_id": call_id,
                "session_id": session_id,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "question": question,
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            audit("approval_required", session_id, {"call_id": call_id, "tool": tool_name})
            return {
                "status": "pending_approval",
                "call_id": call_id,
                "tool_used": tool_name,
                "message": f"Tool '{tool_name}' requires human approval. Use call_id={call_id} to approve.",
            }

        # 4. Execute
        try:
            audit("tool_execute", session_id, {"call_id": call_id, "tool": tool_name})
            result = tool.fn(**tool_input)
            audit("tool_success", session_id, {"call_id": call_id, "tool": tool_name})
            return {
                "status": "success",
                "call_id": call_id,
                "tool_used": tool_name,
                "result": result,
            }
        except Exception as e:
            audit("tool_error", session_id, {"call_id": call_id, "tool": tool_name, "error": str(e)})
            return {"status": "error", "call_id": call_id, "tool_used": tool_name, "message": str(e)}


def approve_pending(call_id: str, approved: bool, approver: str = "human") -> dict:
    """Approve or reject a pending tool call."""
    entry = _pending.get(call_id)
    if not entry:
        return {"status": "error", "message": "call_id not found in pending queue"}
    entry["status"] = "approved" if approved else "rejected"
    entry["approver"] = approver
    entry["decided_at"] = datetime.now(timezone.utc).isoformat()
    audit(
        "approval_decision",
        entry["session_id"],
        {"call_id": call_id, "decision": entry["status"], "approver": approver},
    )
    return entry


def get_pending() -> list[dict]:
    return [v for v in _pending.values() if v["status"] == "pending"]
