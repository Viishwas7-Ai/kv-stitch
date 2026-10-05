"""Workflows: module sequences an app knows people use, warmed into the cache up front.

File format, one workflow per line (blank lines and # comments are ignored):

    web_note_reminder: WEB, NOTES, REMINDER
    clean_downloads:   RECALL, RENAME, RELEVANT
    pc_cleanup:        PCCONTROL, SUGGEST          @granite3.1-moe:3b

The order is the order the modules appear in the prompt. An optional `@model` limits a
workflow to one model; without it the workflow is warmed for whichever model is asked.

For every workflow two prefixes are warmed and pinned:
  * the START   - everything up to and including its last module. A request with the same
                  modules followed by extra ones reuses it exactly; only the extras, the fixed
                  parts after them and the request are computed.
  * the PROMPT  - the full prefix for exactly that workflow (modules, fixed parts, rules),
                  so an exact match is the fastest path.
The app's own builder makes the pieces, so every condition it applies (which rules are kept,
optional sections, lines that depend on other modules) is part of what gets cached. When an
extra module would change the text of an earlier one, the cached text simply no longer
matches and the longest start that still matches is used instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Workflow:
    name: str
    modules: list[str]
    model: str | None = None


_LINE = re.compile(r"^\s*([\w\-. ]+?)\s*:\s*([^@#]+?)\s*(?:@\s*(\S+))?\s*(?:#.*)?$")


def parse_workflows(text: str) -> list[Workflow]:
    out, seen = [], set()
    for no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            raise ValueError(f"workflows line {no}: expected 'name: MOD_A, MOD_B [@model]', got {raw!r}")
        name = m.group(1).strip()
        mods = [x.strip() for x in m.group(2).split(",") if x.strip()]
        if not mods:
            raise ValueError(f"workflows line {no}: no modules")
        if name in seen:
            raise ValueError(f"workflows line {no}: duplicate workflow name {name!r}")
        seen.add(name)
        out.append(Workflow(name, mods, m.group(3)))
    return out


def load_workflows(path: str) -> list[Workflow]:
    with open(path) as f:
        return parse_workflows(f.read())
