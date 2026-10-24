"""Prompt echo (PH402 local, opt-in): the next request carries the ids the model generated, not a re-tokenization.

A client sends the conversation back as text every turn.  The model sometimes writes a piece of text with tokens the
tokenizer would not choose for it (a non-canonical split: "单一" + "起源" where the BPE writes "单" + "一起" + "源").
Re-tokenized, the reply no longer matches what the engine holds, and the engine - whose recurrent (GDN) state cannot
step back to an arbitrary token, and which keeps no checkpoint inside a reply - goes back to the turn checkpoint
before the reply and reads the whole reply again (2026-10-13: 5 of 12 turns of one session, 3,407 / 3,179 / 1,874
tokens; about 1 in 9K generated tokens, so long replies are hit most).

For each request that ended cleanly on <|im_end|> this keeps the text the engine saw (the rendered prompt, with the
literal-tag marks of mark_think_literals, plus the decoded reply) next to the ids it consumed (prompt ids + generated
ids).  A later prompt whose rendered text agrees with a kept one up to some <|im_end|> takes the kept ids up to that
<|im_end|> and tokenizes only the rest.  <|im_end|> is a control token the tokenizer splits on before any BPE, so
cutting there leaves the tokenization on both sides as it was; text that disagrees falls back to the previous
<|im_end|> (at worst: tokenized as without the echo).  A prompt whose turns were all written canonically gets exactly
the ids it got before.

The kept entries must outlive what the engine keeps: an entry dropped while the engine still holds that conversation
turns the next turn back into canonical ids and the engine reads from the first echoed turn on.  So the entry a
request's head came from is replaced by the request's own entry (not kept beside it), entries are dropped least
recently USED first, and their number follows the engine's parked-conversation slots.  Echo stays off where the
serve can lose its entries while the engine keeps the conversation (parked files, slot restore).
"""
from __future__ import annotations

import bisect
import threading
from array import array

IM_END_TEXT = "<|im_end|>"


def common_prefix_len(a: str, b: str) -> int:
    """Length of the common prefix of two (long) strings: chunked slice compares, then a binary search."""
    n, lo, step = min(len(a), len(b)), 0, 1 << 15
    while lo < n and a[lo:lo + step] == b[lo:lo + step]:
        lo += step
    if lo >= n:
        return n
    hi = min(lo + step, n)
    while lo < hi:                                      # the first index in [lo, hi) where they differ
        mid = (lo + hi) // 2
        if a[lo:mid + 1] == b[lo:mid + 1]:
            lo = mid + 1
        else:
            hi = mid
    return lo


class Entry:
    __slots__ = ("text", "ids", "tcut", "icut", "gen_at")

    def __init__(self, text: str, ids: array, tcut: list[int], icut: list[int], gen_at: int):
        self.text, self.ids, self.tcut, self.icut, self.gen_at = text, ids, tcut, icut, gen_at


class Hit:
    """What lookup() found: `chars` of the prompt text are covered by `ids`; `full` when the whole entry matched (the
    engine then holds all of `ids`); `entry` is replaced by the request's own entry when that is recorded."""
    __slots__ = ("chars", "ids", "full", "entry", "differs_in_reply", "context")

    def __init__(self, chars=0, ids=(), full=False, entry=None, differs_in_reply=None, context=None):
        self.chars, self.ids, self.full, self.entry = chars, list(ids), full, entry
        self.differs_in_reply = differs_in_reply        # (char offset into the reply) when the text left the newest
        self.context = context                          # entry inside its last reply; context: (kept, sent) excerpts


class PromptEcho:
    def __init__(self, im_end_id: int, max_entries: int = 32, max_tokens: int = 8_000_000, log=None):
        self.im_end, self.max_entries, self.max_tokens = im_end_id, max(1, max_entries), max(1, max_tokens)
        self.entries: list[Entry] = []                  # least recently used first
        self.lock = threading.Lock()
        self.log = log or (lambda msg: None)
        self.stats = {"full": 0, "partial": 0, "none": 0, "recorded": 0, "evicted": 0}
        self.refused: dict[str, int] = {}               # why a finished request was not kept

    def _cuts(self, text: str, ids) -> tuple[list[int], list[int]] | None:
        """(text offsets, id offsets) just after every <|im_end|>; None when they do not pair up one to one."""
        tpos, i = [], text.find(IM_END_TEXT)
        while i >= 0:
            tpos.append(i + len(IM_END_TEXT))
            i = text.find(IM_END_TEXT, i + 1)
        ipos = [j + 1 for j, t in enumerate(ids) if t == self.im_end]
        return (tpos, ipos) if len(tpos) == len(ipos) else None

    def lookup(self, text: str) -> Hit:
        with self.lock:
            best, best_i, best_n, best_j = None, 0, 0, -1
            for e in self.entries:
                n = len(e.text) if text.startswith(e.text) else common_prefix_len(text, e.text)
                j = bisect.bisect_right(e.tcut, n) - 1
                if j >= 0 and e.icut[j] > best_i:
                    best, best_i, best_n, best_j = e, e.icut[j], n, j
            if best is None:
                self.stats["none"] += 1
                return Hit()
            full = best_j == len(best.tcut) - 1
            self.stats["full" if full else "partial"] += 1
            self.entries.remove(best)                   # most recently used last
            self.entries.append(best)
            hit = Hit(best.tcut[best_j], best.ids[:best_i], full, best)
            if not full and best_n >= best.gen_at:
                # the new text agrees with the newest entry past its prompt and differs inside the reply it kept
                a = max(best.gen_at, best_n - 24)
                hit.differs_in_reply = best_n - best.gen_at
                hit.context = (best.text[a:best_n + 24], text[a:best_n + 24])
            return hit

    def refuse(self, why: str) -> None:
        with self.lock:
            self.refused[why] = self.refused.get(why, 0) + 1

    def record(self, text: str, ids: list[int], gen_at: int, source: Entry | None = None) -> bool:
        """Keep `ids` (the prompt ids and the generated ids, ending with <|im_end|>) for `text` (the rendered prompt
        and the decoded reply).  `source`: the entry this request's echoed head came from - replaced by this one."""
        c = self._cuts(text, ids)
        if c is None or not ids or ids[-1] != self.im_end or not text.endswith(IM_END_TEXT):
            self.refuse("unpaired <|im_end|>")
            return False
        entry = Entry(text, array("i", ids), c[0], c[1], gen_at)
        with self.lock:
            self.entries = [e for e in self.entries if e is not source and not text.startswith(e.text)]
            self.entries.append(entry)
            total = sum(len(e.ids) for e in self.entries)
            while len(self.entries) > 1 and (len(self.entries) > self.max_entries or total > self.max_tokens):
                old = self.entries.pop(0)
                total -= len(old.ids)
                self.stats["evicted"] += 1
                self.log(f"[strata] prompt echo: dropped the least recently used entry ({len(old.ids)} ids; "
                         f"{len(self.entries)} kept, {total} ids) - if the engine still holds that conversation, "
                         "its next turn is read from the first echoed turn")
            self.stats["recorded"] += 1
        return True

    # Kept over an engine restart (2026-10-24): parked conversations that come back after a restart (STRATA_PARK_DIR,
    # PARKSAVE / PARKLOAD) hold the echoed ids, so the entries must come back with them - else the next turn is
    # tokenized canonically and the engine reads from the first non-canonical turn on.  A stale entry is harmless: its
    # ids are a tokenization of the very text it matches.
    EXPORT_VERSION = 1

    def export(self, now: float | None = None) -> bytes:
        """The entries (least recently used first) as one zlib-compressed JSON blob."""
        import base64, json, time, zlib
        with self.lock:
            entries = [{"text": e.text, "ids": base64.b64encode(e.ids.tobytes()).decode("ascii"), "gen_at": e.gen_at}
                       for e in self.entries]
        doc = {"v": self.EXPORT_VERSION, "im_end": self.im_end, "saved_unix": time.time() if now is None else now,
               "itemsize": array("i").itemsize, "entries": entries}
        return zlib.compress(json.dumps(doc, ensure_ascii=False).encode("utf-8"), 1)

    def import_blob(self, blob: bytes, max_age_s: float = 0, now: float | None = None) -> tuple[int, str]:
        """Add the entries of an export() (checked: version, <|im_end|> id, age, every entry's cuts pairing up as
        record() requires) in front of the ones kept now.  -> (entries added, why none were when 0)."""
        import base64, json, time, zlib
        try:
            doc = json.loads(zlib.decompress(blob).decode("utf-8"))
        except (ValueError, zlib.error) as e:
            return 0, f"unreadable ({e})"
        if not isinstance(doc, dict) or doc.get("v") != self.EXPORT_VERSION or doc.get("im_end") != self.im_end \
                or doc.get("itemsize") != array("i").itemsize:
            return 0, "another format or tokenizer"
        age = (time.time() if now is None else now) - float(doc.get("saved_unix") or 0)
        if max_age_s and age > max_age_s:
            return 0, f"saved {age / 3600:.1f} h ago (older than {max_age_s / 3600:.0f} h)"
        fresh: list[Entry] = []
        for d in doc.get("entries") or []:
            try:
                text, gen_at = d["text"], int(d["gen_at"])
                ids = array("i")
                ids.frombytes(base64.b64decode(d["ids"]))
            except (KeyError, TypeError, ValueError):
                continue
            c = self._cuts(text, ids)
            if c is None or not ids or ids[-1] != self.im_end or not text.endswith(IM_END_TEXT) or not 0 <= gen_at <= len(text):
                continue
            fresh.append(Entry(text, ids, c[0], c[1], gen_at))
        if not fresh:
            return 0, "no usable entries"
        with self.lock:
            self.entries = fresh + [e for e in self.entries if all(e.text != f.text for f in fresh)]
            while len(self.entries) > self.max_entries:
                self.entries.pop(0)
        return len(fresh), ""

    def summary(self) -> dict:
        with self.lock:
            return {**self.stats, "entries": len(self.entries), "ids": sum(len(e.ids) for e in self.entries),
                    "refused": dict(self.refused)}
