"""Patch 010 entrypoint — build_system_prompt_for_payload (stage-2 v3, 2026-04-29).

v3 reproduces hermes' real production memory pipeline: production injects
prefetch_all output into the user message via build_memory_context_block,
not into the system prompt. v2 missed this — it only called provider.prefetch
once and dumped result into system_prompt.

Process-singleton session binding (ultra-review 2026-05-11):
  The MemoryManager is built lazily on first call and bound to the fixed
  sid ``debug-build-singleton`` via ``initialize_all``. Honcho, Hindsight
  and OpenViking persist their per-session state from that init call, so
  threading a caller-supplied ``session_id`` into per-call ``prefetch`` is
  effectively a no-op — providers continue to read/write the
  ``debug-build-singleton`` session's state. The previous signature
  accepted ``session_id`` and threaded it through anyway, which silently
  failed to isolate eval callers from one another. The parameter is
  removed in this version; if you need per-session isolation, use the
  real ``/v1/runs`` path or run a fresh process per session.

Returns:
  system_prompt:           Memory provider system_prompt_blocks (tool announcements,
                           ~700 chars).
  user_memory_context:     The fenced <memory-context>...</memory-context> block
                           hermes-agent injects at the end of the user message.
                           This is where OV Session Summary, HS Deductive/Inductive
                           Observations, Honcho Peer/Identity Cards actually live.
  raw_prefetch:            Concatenated prefetch_all() output before fencing.
  memory_blocks:           Per-layer prefetch result, for Daemon-Injection Drop.
  raw_daemon:              Same as memory_blocks (alias).
  token_count:             len(system_prompt) + len(user_memory_context).

Implementation notes:
  - We trigger queue_prefetch_all + 2.0s wait before prefetch_all, since the
    endpoint runs in an ephemeral context with no prior turn to prime caches.
  - Honcho often returns synchronously even without queue priming.
  - HS first-turn-sync-recall (Patch 004) often hits 4s timeout in cold paths.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_mgr_lock = threading.Lock()
_mgr_singleton = None
_PREFETCH_WAIT_S = 2.0  # let queued background recall finish
_SINGLETON_SID = "debug-build-singleton"


def _get_or_init_manager():
    global _mgr_singleton
    if _mgr_singleton is not None:
        return _mgr_singleton
    with _mgr_lock:
        if _mgr_singleton is not None:
            return _mgr_singleton
        from hermes_cli.config import load_config  # type: ignore
        from agent.memory_manager import MemoryManager  # type: ignore
        from plugins.memory import load_memory_provider  # type: ignore
        from hermes_constants import get_hermes_home  # type: ignore

        cfg = load_config()
        mem = (cfg or {}).get('memory') or {}
        names = mem.get('providers') or []
        if not names:
            single = (mem.get('provider') or '').strip()
            if single:
                names = [single]

        mgr = MemoryManager()
        activated = []
        for name in names:
            try:
                p = load_memory_provider(str(name).strip())
                if p and p.is_available():
                    mgr.add_provider(p)
                    activated.append(p.name)
                else:
                    logger.warning('[debug_build] provider %s not available (skipped)', name)
            except Exception as exc:
                logger.warning('[debug_build] provider %s load failed: %s', name, exc)

        if mgr.providers:
            try:
                mgr.initialize_all(
                    session_id=_SINGLETON_SID,
                    platform='api_server',
                    hermes_home=str(get_hermes_home()),
                    agent_context='debug',
                )
            except Exception as exc:
                logger.warning('[debug_build] initialize_all failed: %s', exc)

        logger.info('[debug_build] init done, providers=%s', activated)
        _mgr_singleton = mgr
        return mgr


def build_system_prompt_for_payload(
    *,
    mode: str,
    event_payload: dict,
) -> dict[str, Any]:
    """Build the system-prompt + memory-context payload a real WeChat turn
    would see. Process-singleton — see module docstring for the
    consequences of dropping the ``session_id`` parameter that earlier
    versions accepted but silently ignored.
    """
    if mode != 'weixin_event':
        raise ValueError(f'unsupported mode: {mode!r}')

    query = (
        event_payload.get('content')
        or event_payload.get('text')
        or event_payload.get('message')
        or ''
    )
    if not query:
        raise ValueError('event_payload missing content/text/message')

    sid = _SINGLETON_SID
    mgr = _get_or_init_manager()

    # Two-phase invocation, mimicking production turn-N → turn-N+1
    try:
        mgr.queue_prefetch_all(query, session_id=sid)
    except Exception as exc:
        logger.warning('[debug_build] queue_prefetch_all failed: %s', exc)

    time.sleep(_PREFETCH_WAIT_S)

    try:
        raw_prefetch = mgr.prefetch_all(query, session_id=sid) or ''
    except Exception as exc:
        logger.warning('[debug_build] prefetch_all failed: %s', exc)
        raw_prefetch = ''

    memory_blocks: dict[str, str] = {}
    for p in mgr.providers:
        try:
            memory_blocks[p.name] = p.prefetch(query, session_id=sid) or ''
        except Exception as exc:
            logger.warning('[debug_build] %s prefetch failed: %s', p.name, exc)
            memory_blocks[p.name] = ''

    builtin_block = ''
    try:
        builtin_block = mgr.build_system_prompt() or ''
    except Exception as exc:
        logger.warning('[debug_build] build_system_prompt failed: %s', exc)

    user_memory_context = ''
    try:
        from agent.memory_manager import build_memory_context_block  # type: ignore
        user_memory_context = build_memory_context_block(raw_prefetch) or ''
    except Exception as exc:
        logger.warning('[debug_build] build_memory_context_block failed: %s', exc)

    return {
        'system_prompt': builtin_block,
        'user_memory_context': user_memory_context,
        'raw_prefetch': raw_prefetch,
        'memory_blocks': memory_blocks,
        'raw_daemon': memory_blocks,
        'token_count': len(builtin_block) + len(user_memory_context),
        'truncated_layers': [],
    }
