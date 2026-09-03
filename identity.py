"""
agent/identity.py  —  Non-Human Identity (NHI) for the agent
================================================================
This module gives the AGENT ITSELF an identity, using the exact mental
model from Section 1's Auth0 setup — just with the agent on the other
end of the flow instead of a person.

                    Human user (Section 1)         AI agent (this module)
Who authenticates?  A person, password/passkey     A backend process, client credential
What proves it?     ID token, tied to a person      Access token tied to the agent's client
What limits access? Scopes granted to user's role   Scopes granted to the agent's specific job
How long does it last? A session, until logout      Short-lived, reissued per task

Flow implemented here (same shape as Auth0 client_credentials grant):

  1. PROVISION   — POST /agent/identity/provision
                   Creates a client_id / client_secret pair for one agent
                   deployment (one document session = one agent instance).
                   Human decides the MAX scopes this client is ever allowed
                   to request — this is written down, not implicit.

  2. TOKEN ISSUE — POST /agent/identity/token
                   client_credentials grant: client_id + client_secret +
                   requested scopes -> short-lived access_token (TTL default
                   5 min, write-scoped tokens capped at 2 min). This mirrors
                   Auth0's /oauth/token endpoint with grant_type=client_credentials.

  3. USE         — Every tool call must present a valid, unexpired,
                   correctly-scoped access_token. No token, or an expired
                   one, or a token with the wrong scope => the call is
                   denied before it reaches the tool. This is enforced in
                   PermissionGate.check_token(), not by trusting the caller.

  4. DEPROVISION — POST /agent/identity/deprovision
                   Revokes the client. All outstanding tokens for it are
                   immediately treated as invalid, regardless of their
                   stated expiry. This is the "disabled when they leave"
                   step for a non-human identity.

To point this at REAL Auth0 instead of the local simulation below, set:
    AUTH0_DOMAIN, AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, AUTH0_AUDIENCE
and swap issue_token() to call:
    POST https://{AUTH0_DOMAIN}/oauth/token
    { "grant_type":"client_credentials", "client_id":..., "client_secret":...,
      "audience": AUTH0_AUDIENCE, "scope": "read:documents" }
The rest of this module (expiry checks, scope checks, deprovision-invalidates-
all-tokens) applies unchanged to a real Auth0-issued token.
"""

import hashlib
import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
READ_TOKEN_TTL_SECONDS  = int(os.getenv("AGENT_READ_TOKEN_TTL",  300))   # 5 min
WRITE_TOKEN_TTL_SECONDS = int(os.getenv("AGENT_WRITE_TOKEN_TTL", 120))   # 2 min — tighter for write
ALL_VALID_SCOPES = {"read:documents", "write:documents"}

# Real-Auth0 passthrough config (unused unless AUTH0_DOMAIN is set)
AUTH0_DOMAIN        = os.getenv("AUTH0_DOMAIN", "")
AUTH0_CLIENT_ID_ENV  = os.getenv("AUTH0_CLIENT_ID", "")
AUTH0_CLIENT_SECRET  = os.getenv("AUTH0_CLIENT_SECRET", "")
AUTH0_AUDIENCE       = os.getenv("AUTH0_AUDIENCE", "")


# ── STORES (in-memory; swap for a real datastore in production) ─────────────
_clients: dict[str, dict] = {}   # client_id -> {client_secret_hash, allowed_scopes, status, owner, session_id, created_at, revoked_at}
_tokens:  dict[str, dict] = {}   # access_token -> {client_id, scopes, issued_at, expires_at}


def _hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


# ── 1. PROVISIONING ───────────────────────────────────────────────────────────

def provision_client(session_id: str, allowed_scopes: list[str], owner: str = "system") -> dict:
    """
    Create a new non-human identity for one agent deployment.
    `allowed_scopes` is the MAX this client can ever request — the
    written-down authorization boundary a human is deciding right now.
    """
    invalid = set(allowed_scopes) - ALL_VALID_SCOPES
    if invalid:
        raise ValueError(f"Unknown scopes: {invalid}. Valid: {ALL_VALID_SCOPES}")

    client_id = "agent_" + uuid.uuid4().hex[:16]
    client_secret = secrets.token_urlsafe(24)

    _clients[client_id] = {
        "client_id": client_id,
        "client_secret_hash": _hash_secret(client_secret),
        "allowed_scopes": list(allowed_scopes),
        "status": "active",
        "owner": owner,
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "revoked_at": None,
    }

    log.info("Provisioned agent identity %s (session=%s, scopes=%s)", client_id, session_id, allowed_scopes)
    return {
        "client_id": client_id,
        "client_secret": client_secret,   # shown once, like a real Auth0 client secret
        "allowed_scopes": allowed_scopes,
        "status": "active",
    }


def deprovision_client(client_id: str, reason: str = "session ended") -> dict:
    """
    Retire a non-human identity. All outstanding tokens for this client
    become invalid immediately, even if their stated expiry hasn't passed.
    """
    client = _clients.get(client_id)
    if not client:
        raise ValueError("client_id not found")
    if client["status"] == "revoked":
        return {"client_id": client_id, "status": "already_revoked"}

    client["status"] = "revoked"
    client["revoked_at"] = datetime.now(timezone.utc).isoformat()

    # Invalidate every token issued to this client
    revoked_tokens = [t for t, v in _tokens.items() if v["client_id"] == client_id]
    for t in revoked_tokens:
        del _tokens[t]

    log.info("Deprovisioned agent identity %s (%s) — %d tokens revoked", client_id, reason, len(revoked_tokens))
    return {"client_id": client_id, "status": "revoked", "tokens_revoked": len(revoked_tokens), "reason": reason}


def get_client(client_id: str) -> Optional[dict]:
    return _clients.get(client_id)


def list_clients(session_id: Optional[str] = None) -> list[dict]:
    clients = _clients.values() if session_id is None else [c for c in _clients.values() if c["session_id"] == session_id]
    return [{k: v for k, v in c.items() if k != "client_secret_hash"} for c in clients]


# ── 2. TOKEN ISSUANCE (client_credentials grant) ─────────────────────────────

def issue_token(client_id: str, client_secret: str, requested_scopes: list[str]) -> dict:
    """
    Mirrors Auth0's POST /oauth/token with grant_type=client_credentials.
    The agent authenticates with its OWN credential (not a person's),
    and receives a short-lived, narrowly-scoped access token.
    """
    client = _clients.get(client_id)
    if not client:
        raise PermissionError("Unknown client_id.")
    if client["status"] != "active":
        raise PermissionError("Client is revoked — cannot issue tokens to a deprovisioned identity.")
    if _hash_secret(client_secret) != client["client_secret_hash"]:
        raise PermissionError("Invalid client_secret.")

    requested = set(requested_scopes)
    allowed = set(client["allowed_scopes"])
    over_scoped = requested - allowed
    if over_scoped:
        raise PermissionError(
            f"Requested scopes {over_scoped} exceed this client's authorized scopes {allowed}. "
            f"This is exactly the over-scoped-access misconfiguration — request denied."
        )

    # Write-scoped tokens get a shorter TTL than read-only ones
    ttl = WRITE_TOKEN_TTL_SECONDS if "write:documents" in requested else READ_TOKEN_TTL_SECONDS

    access_token = "at_" + uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=ttl)

    _tokens[access_token] = {
        "client_id": client_id,
        "scopes": list(requested),
        "issued_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
    }

    log.info("Issued token for %s | scopes=%s | ttl=%ss", client_id, requested, ttl)
    return {
        "access_token": access_token,
        "token_type": "Bearer",
        "scope": " ".join(requested),
        "expires_in": ttl,
        "expires_at": expires_at.isoformat(),
    }


# ── 3. VALIDATION (used by every tool call) ──────────────────────────────────

def validate_token(access_token: str, required_scope: str) -> tuple[bool, str]:
    """
    Returns (is_valid, reason). Checks, in order:
      - token exists
      - token not expired
      - underlying client not revoked (covers deprovision-while-token-alive)
      - token carries the scope this specific tool requires
    """
    token = _tokens.get(access_token)
    if not token:
        return False, "Token not found or already revoked."

    expires_at = datetime.fromisoformat(token["expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        del _tokens[access_token]   # expired tokens are cleaned up, not left lying around
        return False, "Token expired. Reissue a new token via /agent/identity/token."

    client = _clients.get(token["client_id"])
    if not client or client["status"] != "active":
        return False, "Underlying identity has been deprovisioned."

    if required_scope not in token["scopes"]:
        return False, f"Token lacks required scope '{required_scope}'. Token has: {token['scopes']}."

    return True, "ok"


def token_info(access_token: str) -> Optional[dict]:
    token = _tokens.get(access_token)
    if not token:
        return None
    expires_at = datetime.fromisoformat(token["expires_at"])
    seconds_left = max(0, int((expires_at - datetime.now(timezone.utc)).total_seconds()))
    return {**token, "seconds_remaining": seconds_left, "is_expired": seconds_left == 0}
