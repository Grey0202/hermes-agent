"""Tests for the observability/fallback-alert plugin.

Covers the fork patch-029 mid-session detection, the 2026-07-08 cold-start
amendment (alert when a session's first API call is already on a configured
fallback provider/model), and the 2026-07-08b per-session primary attribution
+ two-tier (high/low) alerting with low-tier daily dedupe.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_INIT = REPO_ROOT / "plugins" / "observability" / "fallback-alert" / "__init__.py"

PRIMARY = ("anthropic", "claude-fable-5")
FALLBACK = ("custom:deepseek", "deepseek-v4-pro")
# The real per-job primary of the 2026-07-08 incident cron (A股收盘报).
GLM_PRIMARY = ("glm-coding", "glm-5.2")


def _load_module():
    # The plugin directory is hyphenated ("fallback-alert"), so it cannot be
    # imported via importlib.import_module — load it from its __init__.py path.
    spec = importlib.util.spec_from_file_location("fallback_alert_under_test", PLUGIN_INIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def plugin(monkeypatch):
    mod = _load_module()
    # Make _credentials() resolve so the hook reaches its decision path.
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "123456")
    for name in (
        "FALLBACK_ALERT_TELEGRAM_BOT_TOKEN",
        "FALLBACK_ALERT_TELEGRAM_CHAT_ID",
        "FALLBACK_ALERT_FALLBACK_MODELS",
        "FALLBACK_ALERT_THROTTLE_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    sent = []
    monkeypatch.setattr(
        mod,
        "_send_telegram",
        lambda token, chat_id, text: (sent.append((token, chat_id, text)) or True),
    )
    mod._reset_state_for_tests()
    mod._configure_for_tests(PRIMARY, fallback_pairs={FALLBACK}, fallback_models={FALLBACK[1]})
    return mod, sent


@pytest.fixture
def plugin_cron(monkeypatch, tmp_path):
    """Like ``plugin`` but with an isolated HERMES_HOME holding a cron jobs.json
    so cron-session primary attribution + job-name resolution can be exercised."""
    mod = _load_module()
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "123456")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in (
        "FALLBACK_ALERT_TELEGRAM_BOT_TOKEN",
        "FALLBACK_ALERT_TELEGRAM_CHAT_ID",
        "FALLBACK_ALERT_FALLBACK_MODELS",
        "FALLBACK_ALERT_THROTTLE_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    cron_dir = tmp_path / "cron"
    cron_dir.mkdir()
    (cron_dir / "jobs.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "c3cb011360a3",
                        "name": "A股收盘报",
                        "model": "glm-5.2",
                        "provider": "glm-coding",
                        "schedule": {"kind": "cron", "expr": "35 15 * * 1-5"},
                    }
                ],
                "updated_at": "2026-07-08T00:00:00Z",
            }
        )
    )

    sent = []
    monkeypatch.setattr(
        mod,
        "_send_telegram",
        lambda token, chat_id, text: (sent.append((token, chat_id, text)) or True),
    )
    mod._reset_state_for_tests()
    # Global default primary is Anthropic; fallback is deepseek — same as prod.
    mod._configure_for_tests(PRIMARY, fallback_pairs={FALLBACK}, fallback_models={FALLBACK[1]})
    return mod, sent


def _call(mod, session_id, provider, model, **extra):
    mod.on_post_api_request(session_id=session_id, provider=provider, model=model, **extra)


def test_cold_start_first_call_is_fallback_alerts(plugin):
    # (a) auth dead before call #1 -> first observed call is the fallback.
    mod, sent = plugin
    _call(mod, "s-cold", *FALLBACK)
    assert len(sent) == 1
    text = sent[0][2]
    assert "cold start" in text
    assert "deepseek-v4-pro" in text
    assert "claude-fable-5" in text  # expected-primary label present


def test_first_primary_then_divergence_alerts(plugin):
    # (b) existing behaviour: primary on call #1, fallback later -> alert.
    mod, sent = plugin
    _call(mod, "s-mid", *PRIMARY)
    assert sent == []
    _call(mod, "s-mid", *FALLBACK)
    assert len(sent) == 1
    assert "cold start" not in sent[0][2]
    assert "deepseek-v4-pro" in sent[0][2]


def test_same_provider_all_along_no_alert(plugin):
    # (c) never diverges -> silent.
    mod, sent = plugin
    _call(mod, "s-ok", *PRIMARY)
    _call(mod, "s-ok", *PRIMARY)
    _call(mod, "s-ok", *PRIMARY)
    assert sent == []


def test_pinned_nonfallback_first_call_no_alert(plugin):
    # Regression guard: cron sessions pinned to a non-default, non-fallback
    # provider (glm) must NOT trigger a cold-start alert.
    mod, sent = plugin
    _call(mod, "s-glm", "custom:glm", "glm-5.2")
    _call(mod, "s-glm", "custom:glm", "glm-5.2")
    assert sent == []


def test_primary_model_variant_first_call_no_alert(plugin):
    # A genuine anthropic session on a non-default model (opus vs the config
    # default fable) is not a fallback -> no cold-start alert.
    mod, sent = plugin
    _call(mod, "s-opus", "anthropic", "claude-opus-4-8")
    assert sent == []


def test_throttle_suppresses_second_alert(plugin):
    mod, sent = plugin
    _call(mod, "s-th", *PRIMARY)
    _call(mod, "s-th", *FALLBACK)  # alert 1
    _call(mod, "s-th", "custom:other", "other-model")  # throttled
    assert len(sent) == 1


def test_cold_start_disabled_when_no_config(monkeypatch):
    # Safe degradation: with no configured fallback set, cold-start detection
    # is off and behaviour matches the pre-amendment plugin (no alert).
    mod = _load_module()
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "1")
    monkeypatch.delenv("FALLBACK_ALERT_FALLBACK_MODELS", raising=False)
    sent = []
    monkeypatch.setattr(mod, "_send_telegram", lambda *a: (sent.append(a) or True))
    mod._reset_state_for_tests()
    mod._configure_for_tests(None, fallback_pairs=set(), fallback_models=set())
    _call(mod, "s-nocfg", *FALLBACK)
    assert sent == []


def test_load_config_expectations_from_yaml(tmp_path, monkeypatch):
    mod = _load_module()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "model:\n"
        "  default: claude-fable-5\n"
        "  provider: anthropic\n"
        "fallback_providers:\n"
        "  - provider: custom:deepseek\n"
        "    model: deepseek-v4-pro\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("FALLBACK_ALERT_FALLBACK_MODELS", raising=False)
    mod._load_config_expectations()
    assert mod._EXPECTED_PRIMARY == ("anthropic", "claude-fable-5")
    assert ("custom:deepseek", "deepseek-v4-pro") in mod._FALLBACK_PAIRS
    assert "deepseek-v4-pro" in mod._FALLBACK_MODELS
    assert mod._is_cold_start_fallback(("custom:deepseek", "deepseek-v4-pro")) is True
    assert mod._is_cold_start_fallback(("anthropic", "claude-fable-5")) is False
    assert mod._is_cold_start_fallback(("custom:glm", "glm-5.2")) is False


# ── 2026-07-08b: per-session attribution + two-tier alerting ────────────────


def test_cron_cold_start_glm_primary_is_low_tier(plugin_cron):
    # The 2026-07-08 incident: cron A股收盘报 (glm-coding/glm-5.2) 429'd and its
    # first observed call is already the deepseek fallback. Expected primary
    # must be the real per-job glm value (NOT the global anthropic default),
    # and the message must be the calm low-tier variant naming the job.
    mod, sent = plugin_cron
    _call(
        mod,
        "cron_c3cb011360a3_20260708_153534",
        *FALLBACK,
        platform="cron",
        finish_reason="tool_calls",
    )
    assert len(sent) == 1
    text = sent[0][2]
    # Real per-job primary, not the global default.
    assert "glm-coding/glm-5.2" in text
    assert "claude-fable-5" not in text
    assert "claude" not in text
    # Low-tier framing, job name present, Anthropic explicitly disclaimed.
    assert "低优" in text
    assert "A股收盘报" in text
    assert "与 anthropic 无关" in text
    # NOT the alarming high-tier header.
    assert "*Hermes fallback activated*" not in text


def test_anthropic_primary_fallback_is_high_tier(plugin):
    # anthropic-primary session falling back to deepseek keeps the prominent
    # high-tier alert (existing behaviour preserved, explicit two-tier proof).
    mod, sent = plugin
    _call(mod, "s-anthropic", *PRIMARY)
    _call(mod, "s-anthropic", *FALLBACK)
    assert len(sent) == 1
    text = sent[0][2]
    assert "*Hermes fallback activated*" in text
    assert "低优" not in text
    assert "claude-fable-5" in text


def test_low_tier_dedupe_within_day(plugin_cron):
    # Two separate cron runs of the same job on the same day (different session
    # suffixes) must yield at most one low-tier message.
    mod, sent = plugin_cron
    _call(mod, "cron_c3cb011360a3_20260708_153534", *FALLBACK, platform="cron")
    assert len(sent) == 1
    _call(mod, "cron_c3cb011360a3_20260708_160000", *FALLBACK, platform="cron")
    assert len(sent) == 1  # deduped by (job-prefix, primary, day)
