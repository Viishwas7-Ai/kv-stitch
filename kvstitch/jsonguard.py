"""Compact JSON at decoding time, without asking the model.

At every step the highest-scoring token that keeps the output a compact JSON object is taken:
no whitespace outside strings, brackets that match, nothing before the opening `{`, and the
answer ends when that object closes. Inside strings anything goes. The model is never told
about the format, so its prompt (and every cache of it) stays exactly the same.

It is a light check, not a full JSON grammar: it guards whitespace, brackets and the end,
which is what a small model gets wrong when asked to write compact JSON itself.
"""
from __future__ import annotations

import heapq

_OUTSIDE = set('{}[]:,"-+.0123456789eEtrufalsn')   # characters JSON allows outside strings
_CLOSE = {"}": "{", "]": "["}


class JsonGuard:
    def __init__(self, piece, top_k: int = 64):
        self.piece = piece          # token id -> text
        self.top_k = top_k
        self.stack: list[str] = []
        self.in_str = False
        self.esc = False
        self.started = False
        self.done = False

    def _scan(self, text: str, commit: bool) -> bool:
        stack = self.stack if commit else list(self.stack)
        in_str, esc, started, done = self.in_str, self.esc, self.started, self.done
        for ch in text:
            if done:
                return False
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if not started:
                if ch != "{":
                    return False
                started = True
                stack.append("{")
                continue
            if ch not in _OUTSIDE:
                return False
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in _CLOSE:
                if not stack or stack[-1] != _CLOSE[ch]:
                    return False
                stack.pop()
                if not stack:
                    done = True
        if commit:
            self.in_str, self.esc, self.started, self.done = in_str, esc, started, done
        return True

    def pick(self, logits) -> int | None:
        """Best allowed token, or None if none of the top_k is allowed."""
        k = self.top_k if self.started else len(logits)   # the opening brace may rank low
        for t in heapq.nlargest(k, range(len(logits)), key=logits.__getitem__):
            p = self.piece(t)
            if p and self._scan(p, commit=False):
                self._scan(p, commit=True)
                return t
        return None
