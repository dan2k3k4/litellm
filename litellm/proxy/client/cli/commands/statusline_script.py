"""Claude Code status line and Codex Stop hook for auto-routed sessions.

`lite` copies this file verbatim to ~/.litellm/statusline.py and registers it as Claude
Code's `statusLine` command and as Codex's `[[hooks.Stop]]` command, so it must stay
standard-library only and must never import litellm. Claude Code re-runs it on every
status refresh (about every 300ms while typing), so the proxy is asked at most once per
TTL per session and every other refresh is served from a small on-disk cache that holds
only the proxy's answer, never the key.

Claude Code pipes a JSON payload on stdin (session_id, transcript_path, model); the routed
model is the `message.model` of the latest foreground assistant line in the transcript,
which is the proxy's response `model` field. That only names the tier model when the
auto-router deployment sets `return_raw_model_name: true`; otherwise it is the alias the
client requested. Codex pipes its Stop event instead (hook_event_name, session_id) and has
no transcript to read, so the routed model comes from the proxy's session record and the
result is printed as a `systemMessage` for the transcript.

Cost figures come from GET /auto_router/session on the proxy, which reads the per-session
rollup written by the spend flush. That flush is asynchronous, so a turn's cost lands a
second or two after the turn; the cache TTL absorbs it. Any failure degrades to the plain
"Routed to" line, and a session the proxy has not recorded yet is cached as absent for the
same TTL so an unrouted session does not poll on every keystroke. The cache lives in a per-user
directory under the temp root, so users of one shared host never contend for it.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import IO, Final, NamedTuple, Protocol
from urllib.parse import urlencode

SESSION_ENDPOINT: Final = "/auto_router/session"
CACHE_TTL_SECONDS: Final = 5.0
FETCH_TIMEOUT_SECONDS: Final = 3
BAR_WIDTH: Final = 24
BAR_FULL: Final = "\u2588"
BAR_EMPTY: Final = "\u2591"
SEPARATOR: Final = " \u00b7 "
TRANSCRIPT_SCAN_LIMIT_BYTES: Final = 4 * 1024 * 1024
CLAUDE_BASE_URL_ENV_KEYS: Final = ("ANTHROPIC_BASE_URL",)
CLAUDE_API_KEY_ENV_KEYS: Final = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")
CODEX_BASE_URL_ENV_KEYS: Final = ("OPENAI_BASE_URL",)
CODEX_API_KEY_ENV_KEYS: Final = ("OPENAI_API_KEY",)
CODEX_STOP_EVENT: Final = "Stop"
LITELLM_LABEL: Final = "LiteLLM"
RESET: Final = "\033[0m"
BOLD: Final = "\033[1m"
DIM: Final = "\033[90m"
LITELLM_COLOR: Final = "\033[38;2;79;70;229m"
BASELINE_COLOR: Final = "\033[38;2;217;119;87m"
EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
EMPTY_ENV: Final[Mapping[str, str]] = MappingProxyType({})


class Session(NamedTuple):
    """The proxy's record of one auto-routed session, as GET /auto_router/session returns it."""

    router_name: str
    last_model: str
    spend: float
    baseline_spend: float
    baseline_model: str | None


class Credentials(NamedTuple):
    """Where the agent dials and how it authenticates: a key already in hand, or a helper that yields one.

    The helper is Claude Code's own apiKeyHelper command. It is only run on a cache miss, so the
    status line spawns it at most once per TTL rather than on every refresh; `identity` is what the
    cache is keyed by and never contains the key the helper returns.
    """

    base_url: str
    api_key: str
    helper: str = ""

    @property
    def usable(self) -> bool:
        return bool(self.base_url and (self.api_key or self.helper))

    @property
    def identity(self) -> str:
        return self.api_key or f"helper:{hashlib.sha256(self.helper.encode()).hexdigest()}"


class Fetched(NamedTuple):
    """Three outcomes: a session, a definite absence (cacheable), or a failure (not cacheable)."""

    session: Session | None
    definitive: bool


class Fetch(Protocol):
    def __call__(self, credentials: Credentials, session_id: str) -> Fetched: ...


def as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, dict) else EMPTY


def as_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def load_json(raw: bytes | str) -> object:
    try:
        return json.loads(raw)
    except ValueError:
        return None


def resolve_base_url(env: Mapping[str, str], keys: tuple[str, ...]) -> str:
    """The proxy root; OpenAI-style clients carry a trailing /v1 that the control route does not use."""
    raw: Final = next((env[key] for key in keys if env.get(key)), "").strip().rstrip("/")
    return raw.removesuffix("/v1")


def resolve_api_key(env: Mapping[str, str], keys: tuple[str, ...]) -> str:
    return next((env[key] for key in keys if env.get(key)), "").strip()


def claude_credentials(env: Mapping[str, str], config_dir: Path) -> Credentials:
    """Exactly what Claude Code itself dials, so the key-scoped lookup finds the rows it wrote.

    Claude Code resolves its credential in this order: ANTHROPIC_AUTH_TOKEN, then ANTHROPIC_API_KEY
    (from its settings env or the shell), and only then the apiKeyHelper. Nothing else is
    consulted: a `lite` shell variable such as LITELLM_PROXY_API_KEY is not a key Claude Code
    ever sends, so honoring it would query the proxy as a different principal.
    """
    api_key: Final = resolve_api_key(env, CLAUDE_API_KEY_ENV_KEYS)
    helper: Final = "" if api_key else _api_key_helper_command(config_dir)
    return Credentials(resolve_base_url(env, CLAUDE_BASE_URL_ENV_KEYS), api_key, helper)


def codex_credentials(env: Mapping[str, str]) -> Credentials:
    """Exactly what Codex itself dials: the OPENAI_* pair its provider reads, which `lite codex` exports."""
    return Credentials(resolve_base_url(env, CODEX_BASE_URL_ENV_KEYS), resolve_api_key(env, CODEX_API_KEY_ENV_KEYS))


def _api_key_helper_command(config_dir: Path) -> str:
    try:
        raw: Final = (config_dir / "settings.json").read_bytes()
    except OSError:
        return ""
    return as_str(as_mapping(load_json(raw)).get("apiKeyHelper")).strip()


def run_api_key_helper(helper: str, env: Mapping[str, str]) -> str:
    """Run Claude Code's apiKeyHelper the way Claude Code does and return the key it prints."""
    try:
        completed: Final = subprocess.run(
            helper, shell=True, capture_output=True, timeout=FETCH_TIMEOUT_SECONDS, env=env, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.decode(errors="replace").strip() if completed.returncode == 0 else ""


def _transcript_line_model(line: bytes) -> str:
    item: Final = as_mapping(load_json(line))
    if item.get("type") != "assistant" or item.get("isSidechain") is True or item.get("agentId"):
        return ""
    return as_str(as_mapping(item.get("message")).get("model"))


def latest_transcript_model(transcript_path: str) -> str:
    """The served model of Claude Code's latest foreground assistant response, or empty."""
    if not transcript_path:
        return ""
    try:
        with Path(transcript_path).open("rb") as transcript:
            size: Final = transcript.seek(0, os.SEEK_END)
            transcript.seek(max(0, size - TRANSCRIPT_SCAN_LIMIT_BYTES))
            tail: Final = transcript.read()
    except OSError:
        return ""
    return next((model for line in reversed(tail.split(b"\n")) if (model := _transcript_line_model(line))), "")


def model_label(model: str, config_dir: Path) -> str:
    """Claude Code's discovered display name for a model id, else the id without its provider prefix."""
    bare: Final = model.rsplit("/", 1)[-1]
    try:
        raw: Final = (config_dir / "cache" / "gateway-models.json").read_bytes()
    except OSError:
        return bare
    listed: Final = as_mapping(load_json(raw)).get("models")
    if not isinstance(listed, list):
        return bare
    entries: Final = tuple(as_mapping(entry) for entry in listed)
    return next(
        (
            as_str(entry.get("display_name"))
            for entry in entries
            if entry.get("id") in (model, bare) and as_str(entry.get("display_name"))
        ),
        bare,
    )


def baseline_label(model: str, config_dir: Path) -> str:
    """A human name for the counterfactual model: `anthropic/claude-opus-5` reads as `Claude Opus 5`."""
    labelled: Final = model_label(model, config_dir)
    if labelled != model.rsplit("/", 1)[-1]:
        return labelled
    return " ".join(word.capitalize() for word in labelled.replace("-", " ").split())


def fetch_session(credentials: Credentials, session_id: str) -> Fetched:
    """Ask the proxy for the session.

    Any 4xx is a definite answer for this credential (no row, a key that may not call the route,
    an expired login) and is cached for the TTL so the status line does not retry every refresh;
    a 5xx or a transport failure is not, so the next refresh tries again.
    """
    query: Final = urlencode((("session_id", session_id),))
    request: Final = urllib.request.Request(
        f"{credentials.base_url}{SESSION_ENDPOINT}?{query}",
        headers={  # mutable-ok: urllib.request.Request takes a dict
            "Authorization": f"Bearer {credentials.api_key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            raw: Final[bytes] = response.read()
    except urllib.error.HTTPError as error:
        return Fetched(session=None, definitive=400 <= error.code < 500)
    except (urllib.error.URLError, OSError):
        return Fetched(session=None, definitive=False)
    session: Final = _session_from_payload(as_mapping(load_json(raw)))
    return Fetched(session=session, definitive=session is not None)


def _session_from_payload(payload: Mapping[str, object]) -> Session | None:
    router_name: Final = as_str(payload.get("router_name"))
    last_model: Final = as_str(payload.get("last_model"))
    spend: Final = payload.get("spend")
    baseline_spend: Final = payload.get("baseline_spend")
    if not router_name or not last_model:
        return None
    if not isinstance(spend, (int, float)) or not isinstance(baseline_spend, (int, float)):
        return None
    return Session(
        router_name=router_name,
        last_model=last_model,
        spend=float(spend),
        baseline_spend=float(baseline_spend),
        baseline_model=as_str(payload.get("baseline_model")) or None,
    )


def cache_path(cache_dir: Path, credentials: Credentials, session_id: str) -> Path:
    """One entry per (proxy, credential, session): the same session id elsewhere is a different answer."""
    identity: Final = "\n".join((credentials.base_url, credentials.identity, session_id))
    return cache_dir / hashlib.sha256(identity.encode()).hexdigest()


def load_session(
    credentials: Credentials,
    session_id: str,
    cache_dir: Path,
    fetch: Fetch = fetch_session,
    now: Callable[[], float] = time.time,
    env: Mapping[str, str] = EMPTY_ENV,
    run_helper: Callable[[str, Mapping[str, str]], str] = run_api_key_helper,
) -> Session | None:
    """The cached session when fresh, else a fetched one; only definite answers are cached.

    Fresh cache: no helper, no network. Miss: the helper runs once if there is no key in hand, and
    a helper that yields nothing means no fetch and nothing cached, so the next refresh tries again.
    """
    path: Final = cache_path(cache_dir, credentials, session_id)
    cached: Final = _read_cache(path)
    fetched_at: Final = cached.get("fetched_at")
    if isinstance(fetched_at, (int, float)) and now() - fetched_at < CACHE_TTL_SECONDS:
        return _session_from_payload(as_mapping(cached.get("session")))
    api_key: Final = credentials.api_key or run_helper(credentials.helper, env)
    if not api_key:
        return None
    fetched: Final = fetch(credentials._replace(api_key=api_key), session_id)
    if fetched.definitive:
        _write_cache(path, fetched.session, now())
    return fetched.session


NOFOLLOW: Final = getattr(os, "O_NOFOLLOW", 0)


def cache_dir_name() -> str:
    """Per-user on a shared temp root, so one user's private directory never locks out another."""
    return f"litellm-statusline-{os.getuid()}" if hasattr(os, "getuid") else "litellm-statusline"


def _own_private_dir(directory: Path) -> bool:
    """Create the cache directory owner-only and refuse one that is not ours or not private.

    A shared temp root lets another local user pre-create the directory; the hashed file names
    are unguessable, but the directory itself must still be ours before anything is read or
    written under it. Windows has no uids and reports no POSIX mode bits, and its temp directory
    is already per user, so there the directory only has to exist and not be a link.
    """
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        status: Final = directory.lstat()
    except OSError:
        return False
    if not os.path.isdir(directory) or os.path.islink(directory):
        return False
    if not hasattr(os, "getuid"):
        return True
    return status.st_uid == os.getuid() and not status.st_mode & 0o077


def _read_cache(path: Path) -> Mapping[str, object]:
    if not _own_private_dir(path.parent):
        return EMPTY
    try:
        descriptor: Final = os.open(path, os.O_RDONLY | NOFOLLOW)
        with os.fdopen(descriptor, "rb") as handle:
            return as_mapping(load_json(handle.read()))
    except OSError:
        return EMPTY


def _write_cache(path: Path, session: Session | None, fetched_at: float) -> None:
    entry: Final = session._asdict() if session else None
    body: Final = json.dumps({"fetched_at": fetched_at, "session": entry})  # mutable-ok: json.dumps takes a dict
    if not _own_private_dir(path.parent):
        return
    try:
        descriptor: Final = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(body)
    except OSError:
        return


def _bar(fraction: float, color: str, width: int, use_color: bool) -> str:
    filled: Final = round(max(0.0, min(1.0, fraction)) * width)
    if not use_color:
        return BAR_FULL * filled + BAR_EMPTY * (width - filled)
    return f"{color}{BAR_FULL * filled}{DIM}{BAR_EMPTY * (width - filled)}{RESET}"


def render(model: str, session: Session | None, config_dir: Path, use_color: bool, bar_width: int = BAR_WIDTH) -> str:
    """The status text: one "Routed to" line, plus the savings header and cost bars when the proxy has them."""

    def paint(code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if use_color else text

    routed: Final = paint(BOLD, f"Routed to: {model}")
    if session is None:
        return routed
    header: Final = f"{session.router_name}{SEPARATOR}{routed}"
    if session.baseline_model is None or session.baseline_spend <= 0:
        return header
    reference: Final = baseline_label(session.baseline_model, config_dir)
    pct: Final = (session.baseline_spend - session.spend) / session.baseline_spend * 100
    delta: Final = paint(LITELLM_COLOR, f"{'-' if pct >= 0 else '+'}{abs(round(pct))}% vs {reference}")
    peak: Final = max(session.spend, session.baseline_spend)
    label_width: Final = max(len(LITELLM_LABEL), len(reference))
    rows: Final = (
        (LITELLM_LABEL, session.spend, LITELLM_COLOR),
        (reference, session.baseline_spend, BASELINE_COLOR),
    )
    lines: Final = (
        f"{paint(DIM, label.ljust(label_width))} {_bar(amount / peak, color, bar_width, use_color)} "
        f"{paint(DIM, f'${amount:.2f}')}"
        for label, amount, color in rows
    )
    return "\n".join((f"{header}  {delta}", *lines))


def color_enabled(env: Mapping[str, str]) -> bool:
    return env.get("NO_COLOR") is None and env.get("TERM", "") not in ("", "dumb")


def status_line(
    payload: Mapping[str, object], env: Mapping[str, str], config_dir: Path, cache_dir: Path, fetch: Fetch
) -> str:
    """Claude Code mode: the transcript names the routed model; the proxy adds the savings when reachable."""
    fallback: Final = as_str(as_mapping(payload.get("model")).get("display_name"))
    served: Final = latest_transcript_model(as_str(payload.get("transcript_path")))
    if not served:
        return fallback or "claude"
    label: Final = model_label(served, config_dir)
    session_id: Final = as_str(payload.get("session_id"))
    credentials: Final = claude_credentials(env, config_dir)
    if not session_id or not credentials.usable:
        return render(label, None, config_dir, color_enabled(env))
    session: Final = load_session(credentials, session_id, cache_dir, fetch, env=env)
    return render(label, session, config_dir, color_enabled(env))


def codex_stop_message(
    payload: Mapping[str, object], env: Mapping[str, str], config_dir: Path, cache_dir: Path, fetch: Fetch
) -> str:
    """Codex mode: nothing until the proxy has recorded the session, then a systemMessage with the cost block.

    The Stop hook runs once per turn, so there is nothing to throttle and the cache is not used:
    a first turn that finds no record yet must not hide the record the next turn will find.
    """
    session_id: Final = as_str(payload.get("session_id"))
    credentials: Final = codex_credentials(env)
    if not session_id or not credentials.usable:
        return ""
    session: Final = fetch(credentials, session_id).session
    if session is None:
        return ""
    text: Final = render(model_label(session.last_model, config_dir), session, config_dir, use_color=False)
    return json.dumps({"systemMessage": f"\n{text}"})  # mutable-ok: json.dumps takes a dict


def run(stdin: IO[str], stdout: IO[str], env: Mapping[str, str], fetch: Fetch = fetch_session) -> None:
    """Never break the agent: a failure renders each mode's own quiet fallback.

    Claude Code gets the model label it already knows; Codex gets no output at all, which is the
    same silent miss as an unrecorded session, never a bare string it would reject as hook JSON.
    """
    body: Final = as_mapping(load_json(stdin.read()))
    codex: Final = body.get("hook_event_name") == CODEX_STOP_EVENT
    config_dir: Final = Path(env.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
    cache_dir: Final = (
        Path(env.get("TMPDIR") or env.get("TEMP") or env.get("TMP") or tempfile.gettempdir()) / cache_dir_name()
    )
    try:
        text: Final = (
            codex_stop_message(body, env, config_dir, cache_dir, fetch)
            if codex
            else status_line(body, env, config_dir, cache_dir, fetch)
        )
    except Exception:  # noqa: BLE001  # a status line must never break the agent session
        stdout.write("" if codex else as_str(as_mapping(body.get("model")).get("display_name")) or "claude")
        return
    stdout.write(text)


if __name__ == "__main__":
    run(sys.stdin, sys.stdout, os.environ)
