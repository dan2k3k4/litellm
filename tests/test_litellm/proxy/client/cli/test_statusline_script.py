"""The status line script is copied verbatim to the user's machine, so these drive it the way Claude Code
and Codex do: the documented stdin payload, a transcript on disk, and the proxy behind an injected fetch."""

import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from litellm.proxy.client.cli.commands import statusline_script
from litellm.proxy.client.cli.commands.statusline_script import (
    CACHE_TTL_SECONDS,
    Credentials,
    Fetched,
    Session,
    cache_dir_name,
    cache_path,
    claude_credentials,
    codex_credentials,
    latest_transcript_model,
    load_session,
    render,
    run,
)

SESSION_ID = "cf712ab8-4c7c-4d48-ba91-eed54bc2956b"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RECORDED = Session(
    router_name="claude-auto",
    last_model="anthropic/claude-sonnet-5",
    spend=0.14,
    baseline_spend=0.38,
    baseline_model="anthropic/claude-opus-5",
)


def _assistant_line(model: str, **extra: object) -> str:
    return json.dumps({"type": "assistant", "message": {"model": model, "role": "assistant"}, **extra})


@pytest.fixture
def transcript(tmp_path: Path) -> Path:
    path = tmp_path / "session.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}}),
                _assistant_line("claude-haiku-4-5"),
                json.dumps({"type": "user", "message": {"role": "user", "content": "harder"}}),
                _assistant_line("claude-sonnet-5"),
                _assistant_line("claude-haiku-4-5", isSidechain=True),
                _assistant_line("claude-haiku-4-5", agentId="agent-1"),
                json.dumps({"type": "progress", "data": {}}),
            )
        )
        + "\n"
    )
    return path


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "claude"
    (directory / "cache").mkdir(parents=True)
    (directory / "cache" / "gateway-models.json").write_text(
        json.dumps({"models": [{"id": "claude-opus-5", "display_name": "Claude Opus 5"}]})
    )
    return directory


def _payload(transcript: Path, session_id: str = SESSION_ID) -> dict:
    return {
        "session_id": session_id,
        "transcript_path": str(transcript),
        "model": {"id": "claude-auto", "display_name": "claude-auto"},
    }


def _env(tmp_path: Path, config_dir: Path, **extra: str) -> dict[str, str]:
    return {
        "TMPDIR": str(tmp_path / "tmp"),
        "CLAUDE_CONFIG_DIR": str(config_dir),
        "TERM": "dumb",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:4000",
        "ANTHROPIC_AUTH_TOKEN": "sk-virtual",
        **extra,
    }


def _run(payload: object, env: dict[str, str], fetch) -> str:
    out = io.StringIO()
    run(io.StringIO(json.dumps(payload)), out, env, fetch)
    return out.getvalue()


class TestTranscript:
    def test_the_latest_foreground_assistant_line_wins_over_later_sidechain_and_agent_lines(self, transcript):
        assert latest_transcript_model(str(transcript)) == "claude-sonnet-5"

    def test_a_missing_or_empty_transcript_yields_nothing(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        assert latest_transcript_model(str(tmp_path / "missing.jsonl")) == ""
        assert latest_transcript_model(str(empty)) == ""
        assert latest_transcript_model("") == ""


class TestCredentials:
    MIXED = {
        "ANTHROPIC_BASE_URL": "http://anthropic-side:4000",
        "ANTHROPIC_AUTH_TOKEN": "sk-ant",
        "OPENAI_BASE_URL": "http://openai-side:4000/v1/",
        "OPENAI_API_KEY": "sk-openai",
    }

    def test_each_agent_reads_the_pair_it_dials_itself(self, tmp_path):
        # A shell that exports both families must not send Codex's hook to the Anthropic proxy.
        assert claude_credentials(self.MIXED, tmp_path) == Credentials("http://anthropic-side:4000", "sk-ant")
        assert codex_credentials(self.MIXED) == Credentials("http://openai-side:4000", "sk-openai")

    def test_lites_own_shell_variables_are_not_a_credential_either_agent_sends(self, tmp_path):
        # A `lite login` shell exports LITELLM_PROXY_*; Claude Code and Codex never read them, so the
        # status line must not query the proxy as that principal while the agent used another.
        (tmp_path / "settings.json").write_text(json.dumps({"apiKeyHelper": "printf sk-from-helper"}))
        env = {"LITELLM_PROXY_URL": "http://lite:4000/", "LITELLM_PROXY_API_KEY": "sk-lite", **self.MIXED}
        assert claude_credentials(env, tmp_path) == Credentials("http://anthropic-side:4000", "sk-ant", "")
        assert codex_credentials(env) == Credentials("http://openai-side:4000", "sk-openai", "")
        helper_only = {"LITELLM_PROXY_API_KEY": "sk-lite", "ANTHROPIC_BASE_URL": "http://p"}
        assert claude_credentials(helper_only, tmp_path) == Credentials("http://p", "", "printf sk-from-helper")
        assert codex_credentials({}) == Credentials("", "", "")

    def test_claude_code_prefers_the_auth_token_over_a_stray_api_key(self, tmp_path):
        env = {"ANTHROPIC_BASE_URL": "http://p", "ANTHROPIC_API_KEY": "sk-stray", "ANTHROPIC_AUTH_TOKEN": "sk-ours"}
        assert claude_credentials(env, tmp_path).api_key == "sk-ours"

    def test_without_an_env_key_claude_code_carries_the_helper_it_itself_uses(self, tmp_path):
        (tmp_path / "settings.json").write_text(json.dumps({"apiKeyHelper": "printf sk-from-helper"}))
        credentials = claude_credentials({"ANTHROPIC_BASE_URL": "http://p"}, tmp_path)
        assert credentials == Credentials("http://p", "", "printf sk-from-helper")
        assert credentials.usable and "sk-from-helper" not in credentials.identity

    def test_codex_never_carries_a_helper(self, tmp_path):
        (tmp_path / "settings.json").write_text(json.dumps({"apiKeyHelper": "printf sk-from-helper"}))
        assert codex_credentials({"OPENAI_BASE_URL": "http://p/v1"}) == Credentials("http://p", "", "")
        assert not codex_credentials({"OPENAI_BASE_URL": "http://p/v1"}).usable

    def test_the_helper_runs_once_per_ttl_and_its_key_reaches_the_fetch(self, tmp_path):
        runs = []
        fetched_with = []

        def run_helper(helper, env):
            runs.append(helper)
            return "sk-from-helper"

        def fetch(credentials, session_id):
            fetched_with.append(credentials.api_key)
            return Fetched(RECORDED, definitive=True)

        credentials = Credentials("http://p", "", "printf sk-from-helper")
        clock = [100.0]
        for _ in range(3):
            load_session(credentials, SESSION_ID, tmp_path, fetch, now=lambda: clock[0], run_helper=run_helper)
            clock[0] += 1.0
        assert runs == ["printf sk-from-helper"]
        assert fetched_with == ["sk-from-helper"]
        assert "sk-from-helper" not in cache_path(tmp_path, credentials, SESSION_ID).read_text()

    def test_a_helper_that_yields_nothing_means_no_fetch_and_nothing_cached(self, tmp_path):
        def fetch(credentials, session_id):
            raise AssertionError("must not fetch without a key")

        credentials = Credentials("http://p", "", "exit 3")
        assert load_session(credentials, SESSION_ID, tmp_path, fetch, run_helper=lambda h, e: "") is None
        assert not cache_path(tmp_path, credentials, SESSION_ID).exists()

    def test_the_real_helper_is_run_through_the_shell_like_claude_code_does(self, tmp_path):
        from litellm.proxy.client.cli.commands.statusline_script import run_api_key_helper

        assert run_api_key_helper("printf sk-from-helper", {"PATH": "/usr/bin:/bin"}) == "sk-from-helper"
        assert run_api_key_helper("exit 3", {"PATH": "/usr/bin:/bin"}) == ""


class TestSessionCache:
    def test_a_definite_answer_is_served_from_the_cache_within_the_ttl(self, tmp_path):
        calls = []

        def fetch(credentials, session_id):
            calls.append(session_id)
            return Fetched(RECORDED, definitive=True)

        clock = [100.0]
        credentials = Credentials("http://p", "sk")
        first = load_session(credentials, SESSION_ID, tmp_path, fetch, now=lambda: clock[0])
        clock[0] = 100.0 + CACHE_TTL_SECONDS - 1
        second = load_session(credentials, SESSION_ID, tmp_path, fetch, now=lambda: clock[0])
        clock[0] = 100.0 + CACHE_TTL_SECONDS + 1
        third = load_session(credentials, SESSION_ID, tmp_path, fetch, now=lambda: clock[0])
        assert first == second == third == RECORDED
        assert calls == [SESSION_ID, SESSION_ID]

    def test_a_404_is_cached_as_absence_but_a_transport_failure_is_retried(self, tmp_path):
        outcomes = iter((Fetched(None, definitive=False), Fetched(None, definitive=True), Fetched(RECORDED, True)))
        calls = []

        def fetch(credentials, session_id):
            calls.append(session_id)
            return next(outcomes)

        credentials = Credentials("http://p", "sk")
        assert load_session(credentials, SESSION_ID, tmp_path, fetch, now=lambda: 1.0) is None
        assert load_session(credentials, SESSION_ID, tmp_path, fetch, now=lambda: 1.0) is None
        assert load_session(credentials, SESSION_ID, tmp_path, fetch, now=lambda: 1.0) is None
        assert len(calls) == 2

    def test_a_cache_directory_that_is_not_private_is_never_used(self, tmp_path):
        # A shared temp root lets another user pre-create the directory; refuse it rather than write into it.
        calls = []

        def fetch(credentials, session_id):
            calls.append(session_id)
            return Fetched(RECORDED, definitive=True)

        shared = tmp_path / "litellm-statusline"
        shared.mkdir(mode=0o755)
        credentials = Credentials("http://p", "sk")
        for _ in range(2):
            assert load_session(credentials, SESSION_ID, shared, fetch, now=lambda: 1.0) == RECORDED
        assert calls == [SESSION_ID, SESSION_ID]
        assert list(shared.iterdir()) == []

    def test_any_client_error_is_a_definite_answer_and_a_server_error_is_not(self, monkeypatch):
        import urllib.error

        from litellm.proxy.client.cli.commands.statusline_script import fetch_session

        def fail_with(code):
            def opener(request, timeout):
                raise urllib.error.HTTPError(request.full_url, code, "x", {}, None)

            return opener

        for code, definitive in ((403, True), (401, True), (404, True), (502, False)):
            monkeypatch.setattr("urllib.request.urlopen", fail_with(code))
            assert fetch_session(Credentials("http://127.0.0.1:1", "sk"), SESSION_ID) == Fetched(None, definitive)

    def test_the_cache_file_holds_the_proxy_answer_and_never_the_key(self, tmp_path):
        credentials = Credentials("http://p", "sk-secret")
        load_session(credentials, SESSION_ID, tmp_path, lambda c, s: Fetched(RECORDED, True))
        path = cache_path(tmp_path, credentials, SESSION_ID)
        assert "sk-secret" not in written and SESSION_ID not in written if (written := path.read_text()) else False
        assert "sk-secret" not in path.name
        assert json.loads(written)["session"]["baseline_model"] == "anthropic/claude-opus-5"
        assert (path.stat().st_mode & 0o777) == 0o600
        assert (path.parent.stat().st_mode & 0o777) == 0o700

    def test_the_same_session_id_against_another_proxy_or_key_is_not_served_from_the_cache(self, tmp_path):
        answers = iter((Fetched(RECORDED, True), Fetched(RECORDED._replace(spend=9.0), True)))
        first = load_session(Credentials("http://p", "sk-a"), SESSION_ID, tmp_path, lambda c, s: next(answers))
        second = load_session(Credentials("http://p", "sk-b"), SESSION_ID, tmp_path, lambda c, s: next(answers))
        assert first == RECORDED and second is not None and second.spend == 9.0


class TestRender:
    def test_savings_header_and_bars_against_the_routers_baseline(self, config_dir):
        text = render("claude-sonnet-5", RECORDED, config_dir, use_color=False, bar_width=10)
        assert text.splitlines() == [
            "claude-auto · Routed to: claude-sonnet-5  -63% vs Claude Opus 5",
            "LiteLLM       ████░░░░░░ $0.14",
            "Claude Opus 5 ██████████ $0.38",
        ]

    def test_a_session_that_cost_more_than_its_baseline_reads_as_a_plus(self, config_dir):
        dearer = RECORDED._replace(spend=0.50, baseline_spend=0.40)
        assert "+25% vs Claude Opus 5" in render("m", dearer, config_dir, use_color=False)

    def test_without_a_baseline_only_the_routed_line_shows(self, config_dir):
        assert render("m", RECORDED._replace(baseline_model=None), config_dir, False) == "claude-auto · Routed to: m"
        assert render("m", None, config_dir, False) == "Routed to: m"

    def test_color_wraps_the_same_text(self, config_dir):
        colored = render("claude-sonnet-5", RECORDED, config_dir, use_color=True, bar_width=10)
        assert ANSI.sub("", colored) == render("claude-sonnet-5", RECORDED, config_dir, use_color=False, bar_width=10)


class TestClaudeCodeMode:
    def test_the_transcript_names_the_routed_model_and_the_proxy_adds_the_savings(self, tmp_path, transcript, config_dir):
        seen = []

        def fetch(credentials, session_id):
            seen.append((credentials, session_id))
            return Fetched(RECORDED, definitive=True)

        text = _run(_payload(transcript), _env(tmp_path, config_dir), fetch)
        assert text.startswith("claude-auto · Routed to: claude-sonnet-5  -63% vs Claude Opus 5\n")
        assert seen == [(Credentials("http://127.0.0.1:4000", "sk-virtual"), SESSION_ID)]

    def test_an_unrecorded_session_degrades_to_the_routed_line(self, tmp_path, transcript, config_dir):
        assert _run(_payload(transcript), _env(tmp_path, config_dir), lambda c, s: Fetched(None, True)) == (
            "Routed to: claude-sonnet-5"
        )

    def test_without_credentials_the_proxy_is_never_asked(self, tmp_path, transcript, config_dir):
        def fetch(credentials, session_id):
            raise AssertionError("must not fetch")

        env = {k: v for k, v in _env(tmp_path, config_dir).items() if k != "ANTHROPIC_AUTH_TOKEN"}
        assert _run(_payload(transcript), env, fetch) == "Routed to: claude-sonnet-5"

    def test_before_the_first_response_the_payloads_display_name_shows(self, tmp_path, config_dir):
        payload = _payload(tmp_path / "missing.jsonl")
        assert _run(payload, _env(tmp_path, config_dir), lambda c, s: Fetched(RECORDED, True)) == "claude-auto"

    def test_a_discovered_display_name_labels_the_routed_model(self, tmp_path, config_dir):
        path = tmp_path / "t.jsonl"
        path.write_text(_assistant_line("anthropic/claude-opus-5") + "\n")
        assert _run(_payload(path), _env(tmp_path, config_dir), lambda c, s: Fetched(None, True)) == (
            "Routed to: Claude Opus 5"
        )

    def test_the_cache_lands_under_the_platforms_temp_dir(self, tmp_path, transcript, config_dir):
        env = {k: v for k, v in _env(tmp_path, config_dir).items() if k != "TMPDIR"}
        env["TEMP"] = str(tmp_path / "wintemp")
        _run(_payload(transcript), env, lambda c, s: Fetched(RECORDED, True))
        assert (tmp_path / "wintemp" / cache_dir_name()).is_dir()
        assert cache_dir_name().endswith(str(os.getuid()))

    def test_a_crash_falls_back_to_the_model_label_claude_code_already_knows(self, tmp_path, transcript, config_dir):
        def fetch(credentials, session_id):
            raise RuntimeError("boom")

        assert _run(_payload(transcript), _env(tmp_path, config_dir), fetch) == "claude-auto"

    def test_garbage_on_stdin_still_prints_something(self, tmp_path, config_dir):
        out = io.StringIO()
        run(io.StringIO("not json"), out, _env(tmp_path, config_dir), lambda c, s: Fetched(None, True))
        assert out.getvalue() == "claude"


class TestCodexMode:
    def test_the_stop_hook_prints_a_system_message_from_the_proxys_record(self, tmp_path, config_dir):
        env = _env(tmp_path, config_dir, OPENAI_BASE_URL="http://127.0.0.1:4000/v1", OPENAI_API_KEY="sk-codex")
        env = {k: v for k, v in env.items() if not k.startswith("ANTHROPIC_")}
        seen = []

        def fetch(credentials, session_id):
            seen.append(credentials)
            return Fetched(RECORDED, definitive=True)

        out = _run({"hook_event_name": "Stop", "session_id": SESSION_ID, "transcript_path": "/nope"}, env, fetch)
        message = json.loads(out)["systemMessage"]
        assert message.splitlines()[1] == "claude-auto · Routed to: claude-sonnet-5  -63% vs Claude Opus 5"
        assert message.startswith("\n")
        assert seen == [Credentials("http://127.0.0.1:4000", "sk-codex")]

    def test_an_unrecorded_session_prints_nothing_so_codex_shows_no_message(self, tmp_path, config_dir):
        payload = {"hook_event_name": "Stop", "session_id": SESSION_ID}
        assert _run(payload, _env(tmp_path, config_dir), lambda c, s: Fetched(None, True)) == ""

    def test_a_crash_prints_nothing_rather_than_text_codex_would_reject(self, tmp_path, config_dir):
        def fetch(credentials, session_id):
            raise RuntimeError("boom")

        env = _env(tmp_path, config_dir, OPENAI_BASE_URL="http://127.0.0.1:4000/v1", OPENAI_API_KEY="sk-codex")
        assert _run({"hook_event_name": "Stop", "session_id": SESSION_ID}, env, fetch) == ""

    def test_a_turn_right_after_an_unrecorded_one_still_asks_the_proxy(self, tmp_path, config_dir):
        # One hook run per turn: a miss on turn one must not be cached across turn two's fetch.
        answers = iter((Fetched(None, definitive=True), Fetched(RECORDED, definitive=True)))
        payload = {"hook_event_name": "Stop", "session_id": SESSION_ID}
        env = _env(tmp_path, config_dir, OPENAI_BASE_URL="http://127.0.0.1:4000/v1", OPENAI_API_KEY="sk-codex")
        assert _run(payload, env, lambda c, s: next(answers)) == ""
        assert "Routed to: claude-sonnet-5" in json.loads(_run(payload, env, lambda c, s: next(answers)))["systemMessage"]


class TestStandalone:
    def test_the_file_runs_under_a_bare_interpreter_with_no_litellm_on_the_path(self, tmp_path, transcript, config_dir):
        # It is copied verbatim to ~/.litellm/statusline.py, so it must be self-contained.
        script = tmp_path / "statusline.py"
        script.write_bytes(Path(statusline_script.__file__).read_bytes())
        env = {k: v for k, v in _env(tmp_path, config_dir).items() if k != "ANTHROPIC_AUTH_TOKEN"}
        completed = subprocess.run(
            [sys.executable, "-I", str(script)],
            input=json.dumps(_payload(transcript)),
            capture_output=True,
            text=True,
            env=env,
            check=True,
            timeout=30,
        )
        assert completed.stdout == "Routed to: claude-sonnet-5"
