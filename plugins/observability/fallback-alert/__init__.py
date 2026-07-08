"""fallback-alert — Telegram notification when Hermes activates provider fallback.

Adapted from upstream PR #30359 (NousResearch/hermes-agent), fork-local patch 029.

Detects, by comparing (provider, model) seen in successive ``post_api_request``
hook calls within the same session, that Hermes has swapped to a different
provider — the signature of an activated fallback after a primary failure
(429 / 5xx / auth-error).

The plugin records the (provider, model) of the first API call of a session as
that session's primary; any later call with a different (provider, model)
triggers a Telegram message. Throttled per session.

Cold-start detection (fork amendment 2026-07-08)
------------------------------------------------
Mid-session diffing has a blind spot: when the primary auth dies *before* a
session's first API call (the 2026-07-02 and 2026-07-08 incidents), the very
first call is already served by the fallback, so it becomes the recorded
"primary" and no later divergence is ever seen -> no alert fires. To close
this, the plugin now reads the gateway's configured primary and fallback
providers from ``config.yaml`` at registration time. If a session's *first*
observed (provider, model) already matches a configured fallback entry, an
alert fires immediately.

We key the cold-start signal on membership in the configured fallback set
(not merely "differs from the configured primary") on purpose: this
deployment legitimately runs many cron sessions pinned to non-default
providers (e.g. glm-*), and normal human sessions may report a different
Anthropic model (opus vs the config default) than ``model.default``. A
"differs from primary" rule would false-alarm on all of those; matching the
fallback set fires only on genuine failover to the configured backup
(custom:deepseek / deepseek-v4-pro). Exotic tertiary fallbacks are still
caught by the 30-min ``anthropic_cred_watch.py`` Check A backstop.

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
FALLBACK_ALERT_FALLBACK_MODELS    -- comma-separated model names treated as fallback for
                                     cold-start detection; a safe fallback used when
                                     config.yaml cannot be read (empty => cold-start off)
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

# Configured expectations, populated once at register() time from config.yaml.
# ``_EXPECTED_PRIMARY`` is the (provider, model) the gateway is configured to
# use by default; ``_FALLBACK_PAIRS`` / ``_FALLBACK_MODELS`` are the configured
# failover targets used for cold-start detection. All empty/None => cold-start
# detection is disabled (safe degradation).
_EXPECTED_PRIMARY: Optional[Tuple[str, str]] = None
_FALLBACK_PAIRS: set = set()
_FALLBACK_MODELS: set = set()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _config_path() -> str:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return os.path.join(home, "config.yaml")


def _load_config_expectations() -> None:
    """Read primary + fallback providers from config.yaml (once, at register).

    Never raises: on any error the expectations are left empty, which disables
    cold-start detection and preserves the plugin's prior behaviour.
    """
    global _EXPECTED_PRIMARY, _FALLBACK_PAIRS, _FALLBACK_MODELS
    primary: Optional[Tuple[str, str]] = None
    pairs: set = set()
    models: set = set()

    try:
        import yaml  # pyyaml is a hard hermes dependency; import lazily anyway

        with open(_config_path()) as fh:
            cfg = yaml.safe_load(fh) or {}
        model_cfg = cfg.get("model") or {}
        p_provider = str(model_cfg.get("provider") or "").strip()
        p_model = str(model_cfg.get("default") or "").strip()
        if p_provider or p_model:
            primary = (p_provider, p_model)
        for entry in cfg.get("fallback_providers") or []:
            if not isinstance(entry, dict):
                continue
            fp = str(entry.get("provider") or "").strip()
            fm = str(entry.get("model") or "").strip()
            if fp or fm:
                pairs.add((fp, fm))
            if fm:
                models.add(fm)
    except Exception as exc:  # noqa: BLE001 — config read must never break the plugin
        logger.warning("fallback-alert: could not read config.yaml expectations: %s", exc)

    # Env fallback/override: an operator can hard-set the fallback model list
    # (comma-separated) when config.yaml is unavailable or when overriding.
    env_models = _env("FALLBACK_ALERT_FALLBACK_MODELS")
    if env_models:
        for tok in env_models.split(","):
            tok = tok.strip()
            if tok:
                models.add(tok)

    _EXPECTED_PRIMARY = primary
    _FALLBACK_PAIRS = pairs
    _FALLBACK_MODELS = models
    if _debug_enabled():
        logger.info(
            "fallback-alert: expected_primary=%s fallback_pairs=%s fallback_models=%s",
            primary, sorted(pairs), sorted(models),
        )


def _is_cold_start_fallback(current: Tuple[str, str]) -> bool:
    """True when a session's *first* call is already on a configured fallback.

    Membership in the configured fallback set (not "differs from primary") is
    the discriminator — see the module docstring for why.
    """
    if _EXPECTED_PRIMARY is not None and current == _EXPECTED_PRIMARY:
        return False
    if current in _FALLBACK_PAIRS:
        return True
    _, model = current
    return bool(model) and model in _FALLBACK_MODELS


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
    cold_start: bool = False,
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
    if cold_start:
        lines.append("*cold start:* first API call already on fallback (primary never observed)")
        lines.append(f"*expected primary:* `{p_provider}/{p_model}`")
    else:
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
        cold_start = False

        with _STATE_LOCK:
            stored = _PRIMARY_BY_SESSION.get(session_id)
            if stored is None:
                # First call of the session: record it as the baseline for
                # mid-session diffing (unchanged behaviour).
                _PRIMARY_BY_SESSION[session_id] = current
                if not _is_cold_start_fallback(current):
                    if _debug_enabled():
                        logger.info(
                            "fallback-alert: recorded primary %s for session %r",
                            current,
                            session_id,
                        )
                    return
                # ...but if that first call is already on a configured fallback,
                # the primary was never healthy this session and mid-session
                # diffing would never fire — alert now (cold-start fix).
                cold_start = True
                primary = _EXPECTED_PRIMARY or current
            elif stored == current:
                return  # still on baseline — silent
            else:
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
            cold_start=cold_start,
        )
        _send_telegram(token, chat_id, text)
    except Exception as exc:
        logger.warning("fallback-alert: hook handler failed: %s", exc)


def _reset_state_for_tests() -> None:
    """Test helper — clears in-memory state."""
    with _STATE_LOCK:
        _PRIMARY_BY_SESSION.clear()
        _LAST_ALERT_BY_SESSION.clear()


def _configure_for_tests(expected_primary, fallback_pairs=None, fallback_models=None) -> None:
    """Test helper — set configured expectations without reading config.yaml."""
    global _EXPECTED_PRIMARY, _FALLBACK_PAIRS, _FALLBACK_MODELS
    _EXPECTED_PRIMARY = expected_primary
    _FALLBACK_PAIRS = set(fallback_pairs or [])
    _FALLBACK_MODELS = set(fallback_models or [])


def register(ctx) -> None:
    """Plugin entrypoint, called by the Hermes plugin manager on activation."""
    _load_config_expectations()
    ctx.register_hook("post_api_request", on_post_api_request)
