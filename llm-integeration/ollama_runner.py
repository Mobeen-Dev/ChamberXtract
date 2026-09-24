#!/usr/bin/env python3
"""
ollama_runner.py: render the master prompt from config.py, call a local Ollama
model, print the answer and save a full record of the run.

Standard library only (no pip installs).

CLI examples:
    python ollama_runner.py --var task="Summarize in 3 bullets" --file input_text=notes.txt
    python ollama_runner.py --var task="Translate to French" --var input_text="Hello" --stream
    python ollama_runner.py --var-file vars.json --option temperature=0.1 --option num_predict=512

Library use:
    from ollama_runner import run
    rec = run({"task": "Explain buck converters", "input_text": "-"})
    print(rec["response"])
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import config

# Matches {{ name }} with optional inner whitespace; names are identifiers.
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class PromptError(ValueError):
    """Raised when prompt variables are missing or unknown."""


class OllamaError(RuntimeError):
    """Raised for non-recoverable Ollama/API failures."""


class _Retryable(Exception):
    """Internal: connection-phase failure that is safe to retry (nothing was streamed yet)."""


@dataclass
class ChatResult:
    """Parsed outcome of one chat call."""
    content: str
    thinking: str
    done_reason: str
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------- prompts
def placeholders(template: str) -> set[str]:
    """Return the set of placeholder names used in a template."""
    return set(_PLACEHOLDER.findall(template))


def render(template: str, values: dict[str, Any]) -> str:
    """
    Substitute {{name}} placeholders. Single-pass regex substitution, so a value
    that itself contains '{{x}}' is inserted literally and never re-expanded.
    """
    missing = placeholders(template) - values.keys()
    if missing:
        raise PromptError(f"Missing prompt variables: {sorted(missing)}")
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)


def build_prompts(variables: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """
    Merge DEFAULT_VARIABLES with caller values (caller wins), reject unknown names
    (catches typos like --var taks=...), and render both prompts.
    Returns (system_prompt, user_prompt, merged_variables).
    """
    known = placeholders(config.SYSTEM_PROMPT) | placeholders(config.USER_PROMPT_TEMPLATE)
    unknown = set(variables) - known
    if unknown:
        raise PromptError(
            f"Unknown variables {sorted(unknown)}; prompts only use {sorted(known)}"
        )
    merged = {**config.DEFAULT_VARIABLES, **variables}
    return (
        render(config.SYSTEM_PROMPT, merged),
        render(config.USER_PROMPT_TEMPLATE, merged),
        {k: merged[k] for k in sorted(known)},
    )


# ------------------------------------------------------------- Ollama I/O
def _extract_error(body: str) -> str:
    """Pull the 'error' field out of an Ollama JSON error body, else return raw text."""
    try:
        return str(json.loads(body).get("error", body))
    except (json.JSONDecodeError, AttributeError):
        return body.strip() or "(empty response body)"


def _chat_once(payload: dict, stream: bool,
               on_token: Optional[Callable[[str], None]]) -> ChatResult:
    """One HTTP round trip to /api/chat. Only connection-phase errors are retryable."""
    url = config.OLLAMA_HOST.rstrip("/") + "/api/chat"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = urllib.request.urlopen(request, timeout=config.REQUEST_TIMEOUT_S)
    except urllib.error.HTTPError as exc:  # must precede OSError (HTTPError subclasses it)
        detail = _extract_error(exc.read().decode("utf-8", "replace"))
        if exc.code >= 500:
            raise _Retryable(f"HTTP {exc.code}: {detail}") from exc
        raise OllamaError(f"HTTP {exc.code}: {detail}") from exc
    except OSError as exc:  # URLError, timeouts, refused connections
        raise _Retryable(f"cannot reach {url}: {exc}") from exc

    content_parts: list[str] = []
    thinking_parts: list[str] = []
    final: Optional[dict] = None

    try:
        with response:
            # Non-stream replies are a single JSON object; stream replies are NDJSON.
            # Iterating lines handles both uniformly (a one-object body is one line
            # when Ollama emits compact JSON, but parse the whole body if it is not).
            if stream:
                chunks = (json.loads(line) for line in
                          (raw.strip() for raw in response) if line)
            else:
                chunks = iter([json.loads(response.read())])

            for chunk in chunks:
                if "error" in chunk:
                    raise OllamaError(str(chunk["error"]))
                message = chunk.get("message") or {}
                piece = message.get("content") or ""
                think_piece = message.get("thinking") or ""
                if piece:
                    content_parts.append(piece)
                    if on_token:
                        on_token(piece)
                if think_piece:
                    thinking_parts.append(think_piece)
                if chunk.get("done"):
                    final = chunk
    except (OSError, json.JSONDecodeError) as exc:
        raise OllamaError(f"failed while reading response: {exc}") from exc

    if final is None:
        raise OllamaError("response ended before the model reported completion")

    # Ollama reports durations in nanoseconds; convert to seconds for readability.
    def secs(key: str) -> Optional[float]:
        value = final.get(key)
        return round(value / 1e9, 3) if isinstance(value, (int, float)) else None

    eval_count = final.get("eval_count")
    eval_s = secs("eval_duration")
    stats = {
        "prompt_tokens": final.get("prompt_eval_count"),
        "completion_tokens": eval_count,
        "total_s": secs("total_duration"),
        "load_s": secs("load_duration"),
        "eval_s": eval_s,
        "tokens_per_s": round(eval_count / eval_s, 2) if eval_count and eval_s else None,
    }
    return ChatResult(
        content="".join(content_parts),
        thinking="".join(thinking_parts),
        done_reason=str(final.get("done_reason", "")),
        stats=stats,
    )


def chat(system_prompt: str, user_prompt: str, options: dict[str, Any],
         stream: bool = False,
         on_token: Optional[Callable[[str], None]] = None) -> ChatResult:
    """Send the prompts to Ollama with retry on transient connection/5xx failures."""
    payload: dict[str, Any] = {
        "model": config.MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": stream,
        "options": options,
        "keep_alive": config.KEEP_ALIVE,
    }
    if config.THINK is not None:
        payload["think"] = config.THINK

    last: Optional[_Retryable] = None
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            return _chat_once(payload, stream, on_token)
        except _Retryable as exc:
            last = exc
            if attempt < config.MAX_RETRIES:
                delay = config.RETRY_BACKOFF_S * (2 ** attempt)
                print(f"[retry {attempt + 1}/{config.MAX_RETRIES}] {exc}; sleeping {delay:.1f}s",
                      file=sys.stderr)
                time.sleep(delay)
    raise OllamaError(f"giving up after {config.MAX_RETRIES + 1} attempts: {last}") from last


# ------------------------------------------------------------------ saving
def save_record(record: dict[str, Any]) -> Path:
    """
    Write the run record to OUTPUT_DIR as JSON. Written to a temp file first and
    moved into place with os.replace, so a crash never leaves a half-written file.
    """
    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{record['timestamp_utc'].replace(':', '-')}_{record['run_id']}.json"
    tmp_path = final_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, final_path)
    return final_path


def run(variables: dict[str, Any],
        option_overrides: Optional[dict[str, Any]] = None,
        stream: bool = False,
        save: bool = True) -> dict[str, Any]:
    """
    Full pipeline: render prompts -> call model -> (optionally) save.
    Returns the record dict, which includes 'response', 'thinking', 'stats'
    and, if saved, 'saved_to'.
    """
    system_prompt, user_prompt, merged = build_prompts(variables)
    options = {**config.OPTIONS, **(option_overrides or {})}

    started = time.perf_counter()
    result = chat(
        system_prompt, user_prompt, options, stream=stream,
        on_token=(lambda t: print(t, end="", flush=True)) if stream else None,
    )
    elapsed = round(time.perf_counter() - started, 3)

    if result.done_reason == "length":
        print("\n[warning] output hit the num_predict limit and may be truncated "
              "(thinking tokens count toward it)", file=sys.stderr)

    record: dict[str, Any] = {
        "run_id": uuid.uuid4().hex[:8],
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": config.MODEL,
        "host": config.OLLAMA_HOST,
        "think": config.THINK,
        "options": options,
        "variables": merged,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response": result.content,
        "thinking": result.thinking,
        "done_reason": result.done_reason,
        "stats": result.stats,
        "wall_clock_s": elapsed,
    }
    if save:
        record["saved_to"] = str(save_record(record))
    return record


# --------------------------------------------------------------------- CLI
def _parse_kv(items: list[str], flag: str) -> dict[str, str]:
    """Parse ['k=v', ...] into a dict; the split happens on the first '=' only."""
    parsed: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"{flag} expects key=value, got: {item!r}")
        parsed[key.strip()] = value
    return parsed


def _coerce(value: str) -> Any:
    """Interpret an option value as JSON (0.2, 512, true) and fall back to a plain string."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the master prompt against a local Ollama model.")
    parser.add_argument("--var", action="append", default=[], metavar="NAME=VALUE",
                        help="prompt variable (repeatable)")
    parser.add_argument("--file", action="append", default=[], metavar="NAME=PATH",
                        help="prompt variable whose value is read from a UTF-8 text file (repeatable)")
    parser.add_argument("--var-file", metavar="JSON",
                        help="JSON object of prompt variables (overridden by --var/--file)")
    parser.add_argument("--option", action="append", default=[], metavar="NAME=VALUE",
                        help="override a config.OPTIONS entry, e.g. temperature=0.1 (repeatable)")
    parser.add_argument("--stream", action="store_true", help="print tokens as they are generated")
    parser.add_argument("--no-save", action="store_true", help="do not write a record to disk")
    args = parser.parse_args(argv)

    variables: dict[str, Any] = {}
    if args.var_file:
        try:
            loaded = json.loads(Path(args.var_file).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read --var-file: {exc}")
        if not isinstance(loaded, dict):
            raise SystemExit("--var-file must contain a JSON object")
        variables.update(loaded)
    variables.update(_parse_kv(args.var, "--var"))
    for name, path in _parse_kv(args.file, "--file").items():
        try:
            variables[name] = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise SystemExit(f"cannot read file for {name}: {exc}")

    overrides = {k: _coerce(v) for k, v in _parse_kv(args.option, "--option").items()}

    try:
        record = run(variables, overrides, stream=args.stream, save=not args.no_save)
    except (PromptError, OllamaError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.stream:
        print()  # terminate the streamed line
    else:
        print(record["response"])

    s = record["stats"]
    print(f"\n--- {record['model']} | {s.get('completion_tokens')} tokens | "
          f"{s.get('tokens_per_s')} tok/s | {record['wall_clock_s']}s wall ---", file=sys.stderr)
    if "saved_to" in record:
        print(f"saved: {record['saved_to']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
