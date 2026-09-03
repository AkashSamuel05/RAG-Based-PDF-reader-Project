# DocMind Agent · Non-Human Identity Edition

An agentic document assistant where **the agent has its own identity** —
provisioned, scoped, token-bound, and revocable — instead of a standing API
key that outlives the demo.

Ollama (Llama 3, primary) → Claude API (fallback). PDF / DOCX / XLSX / CSV / TXT.

---

## Why this exists

A RAG tool that only returns text is bounded by human review — if it's
wrong, a person reads the wrong sentence and decides what to do. An agent
that can call tools directly changes that: a mistake or a prompt injection
can now cause a real action instead of a wrong sentence on a screen.

This app treats the agent the way you'd treat any other identity with
access:

- **Who provisions it?** Uploading a document provisions a `client_id` /
  `client_secret` pair for that session — visible in the sidebar.
- **What is it authorized to do, written down where?** `allowed_scopes` is
  set at provision time (`read:documents`, `write:documents`) and every
  tool declares the scope it needs. Requesting a scope outside that set is
  rejected at token issuance, not silently allowed.
- **Who deprovisions it, and how would you know if that step was
  skipped?** Click "Deprovision" any time, and every outstanding token for
  that identity dies immediately — visible in the audit log with a
  `tokens_revoked` count.

## Identity flow (mirrors Auth0's client_credentials grant)

```
Upload document
      │
      ▼
PROVISION  ── POST /agent/identity/provision
              creates client_id + client_secret,
              scoped to read:documents / write:documents
      │
      ▼
TOKEN ISSUE ── POST /agent/identity/token
               client authenticates with its OWN secret
               → short-lived access_token
               (300s for read, 120s for write — write is tighter)
               requesting an unauthorized scope = 403, logged
      │
      ▼
USE ────────── every tool call requires Authorization: Bearer <token>
               PermissionGate validates: exists? expired? right scope?
               underlying client still active?
               No valid token → denied before the tool ever runs.
      │
      ▼
DEPROVISION ── POST /agent/identity/deprovision
               revokes the client; every outstanding token for it
               is invalidated immediately, regardless of stated expiry
```

Every step above writes an entry to the audit log (📋 button, top right) —
`session_created`, `token_issued`, `token_check` (ALLOW/DENY + reason),
`injection_detected`, `identity_deprovisioned`.

## Other safety layers

- **Prompt injection sanitizer** — document text is scanned for patterns
  like "ignore previous instructions" or "send an email to" before it
  ever reaches a prompt. Matches are redacted and logged, not silently
  followed.
- **Human approval gate** — `export_summary` (a write action) is tagged
  `requires_approval=True`. Even with a valid write-scoped token, it
  returns `pending_approval` rather than auto-executing.
- **Chunk-based RAG** — the document is split into adjustable chunks
  (250–2000 words, configurable overlap) and only the top-K relevant
  chunks are retrieved per question, with chunk IDs and relevance scores
  shown in the response — you can see exactly what the model saw.

---

## Project structure

```
docmind-agent/
├── main.py               ← FastAPI app: routes, file parsing, AI calls
├── agent/
│   ├── core.py            ← tool registry, injection sanitizer, chunking/retrieval, audit log
│   └── identity.py        ← non-human identity: provision / token / validate / deprovision
├── static/
│   └── index.html          ← full frontend (identity panel, chat, audit log viewer)
├── requirements.txt
├── start.sh
└── README.md
```

**Languages used:** Python (FastAPI) · Bash · HTML/CSS/JavaScript

---

## Quick start

```bash
# 1. Install Ollama (recommended — keeps everything local)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3

# 2. Run
chmod +x start.sh
./start.sh

# 3. Open
http://localhost:8000
```

### Manual setup

```bash
pip install -r requirements.txt
export OLLAMA_MODEL=llama3                 # default
export OLLAMA_URL=http://localhost:11434   # default
export ANTHROPIC_API_KEY=sk-ant-...        # optional fallback
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Using the identity flow from the UI

1. Upload a document — a client_id/secret is provisioned automatically,
   shown in the **Agent Identity** panel.
2. Click **🔓 Get read token** before asking questions (the app also does
   this automatically the first time you ask something).
3. Ask questions, summarize, quiz, extract key terms — all read-scoped.
4. Click **📤 Export** to try a write action — it will request a
   write-scoped token (shorter TTL) and then hold for human approval
   rather than executing immediately.
5. Click **🗑 Deprovision** to revoke the identity and watch every
   outstanding token die — try asking another question afterward and see
   it get denied.
6. Open **📋 Audit Log** any time to see the full decision trail.

## API reference

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/upload` | Upload a document → provisions agent identity |
| `POST` | `/rechunk` | Change chunk size/overlap and re-split |
| `POST` | `/agent/identity/provision` | (Re-)provision an identity with explicit scopes |
| `POST` | `/agent/identity/token` | client_credentials grant → short-lived access token |
| `POST` | `/agent/identity/deprovision` | Revoke identity + all its tokens |
| `GET`  | `/agent/identity/{client_id}` | Inspect an identity's status/scopes |
| `POST` | `/agent/ask` | Ask a question — requires `Authorization: Bearer <token>` |
| `POST` | `/agent/export` | Write action — requires write-scoped token + human approval |
| `GET`  | `/agent/audit` | Full audit trail (optionally filtered by session) |
| `GET`  | `/health` | Ollama/Claude availability check |

### Connecting to real Auth0 instead of the local simulation

`agent/identity.py` implements the same shape as Auth0's
`client_credentials` grant locally, with no external dependency. To point
it at a real Auth0 tenant, set:

```bash
export AUTH0_DOMAIN=your-tenant.auth0.com
export AUTH0_CLIENT_ID=...
export AUTH0_CLIENT_SECRET=...
export AUTH0_AUDIENCE=...
```

and swap `issue_token()` in `agent/identity.py` to call
`POST https://{AUTH0_DOMAIN}/oauth/token` with `grant_type=client_credentials`.
Everything downstream — expiry checks, scope checks, deprovision
invalidating tokens — applies unchanged to a real Auth0-issued token.

---

## Limitations

- Scanned/image-only PDFs aren't supported (no OCR)
- Sessions, identities, and tokens are in-memory — restart clears them
- Retrieval is keyword-based (no embeddings) — fully offline, no vector DB needed
- Local identity simulation is for demonstrating the pattern; swap in real
  Auth0 (or another IdP) for production use
