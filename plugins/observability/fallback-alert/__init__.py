"""fallback-alert — Telegram notification when Hermes activates provider fallback.

Adapted from upstream PR #30359 (NousResearch/hermes-agent), fork-local patch 029.

Detects, by comparing (provider, model) seen in successive ``post_api_request``
hook calls within the same session, that Hermes has swapped to a different
provider — the signature of an activated fallback after a primary failure
(429 / 5xx / auth-error).

The plugin records the (provider, model) of the first API call of a session as
that session's primary; any later call with a different (provider, model)
triggers a Telegram message. Throttled per session.

Fork adaptation
---------------
This deployment already exports the gateway's Telegram bot token and home
channel via ``~/.hermes/.env`` (loaded into os.environ by
``hermes_cli.main.load_hermes_dotenv`` at startup). ``_credentials()`` therefore
falls back to ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_HOME_CHANNEL`` so no separate
secret wiring is needed. The upstream ``FALLBACK_ALERT_TELEGRAM_*`` names still
win when set, for an alternate bot/channel.

Env vars (any of)
-----------------
FALLBACK_ALERT_TELEGRAM_BOT_TOKEN | TELEGRAM_BOT_TOKEN   -- bot token
FALLBACK_ALERT_TELEGRAM_CHAT_ID   | TELEGRAM_HOME_CHANNEL -- target chat id / @channel
FALLBACK_ALERT_THROTTLE_SECONDS   -- min seconds between alerts per session (default 300)
FALLBACK_ALERT_DEBUG              -- ``true`` to log no-op reasons at INFO level
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Module-level state: each entry is per-session.
_PRIMARY_BY_SESSION: Dict[str, Tuple[str, str]] = {}
_LAST_ALERT_BY_SESSION: Dict[str, float] = {}
_STATE_LOCK = threading.Lock()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _debug_enabled() -> bool:
    return _env("FALLBACK_ALERT_DEBUG").lower() in {"1", "true", "yes", "on"}


def _throttle_seconds() -> int:
    try:
        return max(1, int(_env("FALLBACK_ALERT_THROTTLE_SECONDS", "300")))
    except ValueError:
        return 300


def _credentials() -> Optional[Tuple[str, str]]:
    # Prefer the plugin-specific overrides; otherwise reuse the gateway's
    # existing Telegram bot + home-channel env (fork adaptation, see module
    # docstring) so no separate secret wiring is required on this deployment.
    token = _env("FALLBACK_ALERT_TELEGRAM_BOT_TOKEN") or _env("TELEGRAM_BOT_TOKEN")
    chat_id = _env("FALLBACK_ALERT_TELEGRAM_CHAT_ID") or _env("TELEGRAM_HOME_CHANNEL")
    if not token or not chat_id:
        return None
    return token, chat_id


def _send_telegram(token: str, chat_id: str, text: str) -> bool:
    """POST to Telegram Bot API. Never raises. Returns True on success."""
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        body = json.dumps(
            {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            }
        ).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=8) as resp:
            if resp.status >= 300:
                logger.warning(
                    "fallback-alert: telegram returned HTTP %d", resp.status
                )
                return False
            return True
    except urllib.error.HTTPError as exc:
        logger.warning(
            "fallback-alert: telegram HTTPError %d: %s",
            exc.code,
            exc.read()[:200].decode("utf-8", "replace"),
        )
    except Exception as exc:
        logger.warning("fallback-alert: telegram send failed: %s", exc)
    return False


def _format_message(
    *,
    session_id: str,
    platform: str,
    primary: Tuple[str, str],
    current: Tuple[str, str],
    finish_reason: str = "",
) -> str:
    p_provider, p_model = primary
    c_provider, c_model = current
    session_short = (session_id[:24] + "…") if len(session_id) > 24 else session_id
    lines = [
        "*Hermes fallback activated*",
        f"*session:* `{session_short or '<no session>'}`",
    ]
    if platform:
        lines.append(f"*platform:* `{platform}`")
    lines.append(f"*primary:* `{p_provider}/{p_model}`")
    lines.append(f"*now:* `{c_provider}/{c_model}`")
    if finish_reason:
        lines.append(f"*finish_reason:* `{finish_reason}`")
    return "\n".join(lines)


def on_post_api_request(**kwargs) -> None:
    """Hook handler. Fires after each API call regardless of outcome.

    The handler never raises — any error is logged and swallowed so the
    plugin can never crash Hermes' main request loop.
    """
    try:
        creds = _credentials()
        if creds is None:
            if _debug_enabled():
                logger.info("fallback-alert: no credentials configured, skipping")
            return

        session_id = (kwargs.get("session_id") or "").strip()
        provider = (kwargs.get("provider") or "").strip()
        model = (kwargs.get("model") or "").strip()
        if not provider or not model:
            return

        current = (provider, model)
        primary: Optional[Tuple[str, str]] = None

        with _STATE_LOCK:
            stored = _PRIMARY_BY_SESSION.get(session_id)
            if stored is None:
                _PRIMARY_BY_SESSION[session_id] = current
                if _debug_enabled():
                    logger.info(
                        "fallback-alert: recorded primary %s for session %r",
                        current,
                        session_id,
                    )
                return
            if stored == current:
                return  # still on primary — silent
            primary = stored

            now = time.time()
            last = _LAST_ALERT_BY_SESSION.get(session_id, 0.0)
            if (now - last) < _throttle_seconds():
                if _debug_enabled():
                    logger.info(
                        "fallback-alert: throttled (%.0fs since last alert for session %r)",
                        now - last,
                        session_id,
                    )
                return
            _LAST_ALERT_BY_SESSION[session_id] = now

        token, chat_id = creds
        text = _format_message(
            session_id=session_id,
            platform=str(kwargs.get("platform") or ""),
            primary=primary,
            current=current,
            finish_reason=str(kwargs.get("finish_reason") or ""),
        )
        _send_telegram(token, chat_id, text)
    except Exception as exc:
        logger.warning("fallback-alert: hook handler failed: %s", exc)


def _reset_state_for_tests() -> None:
    """Test helper — clears in-memory state."""
    with _STATE_LOCK:
        _PRIMARY_BY_SESSION.clear()
        _LAST_ALERT_BY_SESSION.clear()


def register(ctx) -> None:
    """Plugin entrypoint, called by the Hermes plugin manager on activation."""
    ctx.register_hook("post_api_request", on_post_api_request)
