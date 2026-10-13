"""serve/test_prompt_echo.py - the prompt echo (PH402 local, opt-in): the next request carries the ids the model
generated, not a re-tokenization of their text.  Mock engine, no GPU, no pack.

    python -m unittest serve.test_prompt_echo -v
"""
from __future__ import annotations

import json
import os
import random
import sys
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.prompt_echo import PromptEcho, common_prefix_len  # noqa: E402
from serve.server import (IM_END, ByteTokenizer, MockEngine, PromptIds, Service, StrataEngine,  # noqa: E402
                          prompt_echo_setup, serve)

ROOT = Path(__file__).resolve().parents[1]
AB = 400                                        # the merged id of "ab" (the BPE's choice); the model writes 97, 98


class MergeTokenizer(ByteTokenizer):
    """ByteTokenizer whose encoding writes "ab" as one id: a reply holding 97, 98 is not what re-tokenizing gives."""

    def encode(self, text, parse_special=False, plain=()):
        out, merged, i = super().encode(text, parse_special, plain), [], 0
        while i < len(out):
            if out[i] == 97 and i + 1 < len(out) and out[i + 1] == 98:
                merged.append(AB)
                i += 2
            else:
                merged.append(out[i])
                i += 1
        return merged

    def decode(self, ids, errors="replace"):
        raw = bytearray()
        for t in ids:
            raw += b"ab" if t == AB else self.SPECIALS[t - 256].encode() if t >= 256 else bytes([t])
        return raw.decode("utf-8", errors=errors)


class IdsEngine(MockEngine):
    """Writes the given id lists (one per request, the last repeats) and keeps every prompt it was given."""

    def __init__(self, tok, scripts):
        super().__init__(tok, "x")
        self.id_scripts, self.prompts = scripts, []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        self.last_prompt = list(ids)
        for t in self.id_scripts[min(len(self.prompts) - 1, len(self.id_scripts) - 1)][:max_new]:
            if cancel.is_set():
                return
            yield t


def ids_of(text: bytes, tok) -> list[int]:
    return list(text)


class Rig:
    """A Service over IdsEngine behind the HTTP server; `chat` posts an OpenAI chat request."""

    def __init__(self, scripts, echo=True, tok=None):
        self.tok = tok or MergeTokenizer()
        self.end = self.tok.encode(IM_END, parse_special=True)
        self.engine = IdsEngine(self.tok, [s + self.end if s and s[-1] < 256 else s for s in scripts])
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        with mock.patch.object(StrataEngine, "PARK_DIR", ""):
            prompt_echo_setup(self.svc, {"prompt_echo": echo}, "")
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def chat(self, messages, **kw):
        req = urllib.request.Request(self.base + "/v1/chat/completions",
                                     data=json.dumps({"model": "m", "messages": messages, **kw}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def converse(self, turns, **kw):
        """Send `turns` user messages one after another, each with the history the client got back (reasoning in
        reasoning_content, as DSH sends it)."""
        history = []
        for text in turns:
            history.append({"role": "user", "content": text})
            msg = self.chat(history, **kw)["choices"][0]["message"]
            back = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("reasoning_content"):
                back["reasoning_content"] = msg["reasoning_content"]
            history.append(back)
        return self.engine.prompts


THINK_NONCANON = list(b"let me think: ") + [97, 98] + list(b" is it\n</think>\n\nhello ") + [97, 98] + list(b" world")
THINK_CANON = list(b"let me think: ") + [AB] + list(b" is it\n</think>\n\nhello ") + [AB] + list(b" world")
SECOND = list(b"fine\n</think>\n\nsecond answer")


class Echo(unittest.TestCase):
    def rig(self, scripts, echo=True, tok=None):
        r = Rig(scripts, echo, tok)
        self.addCleanup(r.close)
        return r

    def test_noncanonical_reply_is_carried_as_generated(self):
        r = self.rig([THINK_NONCANON, SECOND])
        p1, p2 = r.converse(["hi", "more"])
        gen1 = THINK_NONCANON + r.end
        self.assertEqual(p2[:len(p1) + len(gen1)], p1 + gen1)       # the engine's own ids: no re-read
        self.assertEqual(r.svc.prompt_echo.stats["full"], 1)

    def test_without_the_echo_the_reply_is_retokenized(self):
        # the control: today's behaviour, which makes the engine read the reply again
        r = self.rig([THINK_NONCANON, SECOND], echo=False)
        p1, p2 = r.converse(["hi", "more"])
        gen1 = THINK_NONCANON + r.end
        self.assertNotEqual(p2[:len(p1) + len(gen1)], p1 + gen1)
        self.assertEqual(p2[:len(p1)], p1)
        self.assertIsNone(r.svc.prompt_echo)

    def test_canonical_conversation_gets_the_same_ids_with_or_without(self):
        on = self.rig([THINK_CANON, SECOND, THINK_CANON]).converse(["hi", "more", "again"])
        off = self.rig([THINK_CANON, SECOND, THINK_CANON], echo=False).converse(["hi", "more", "again"])
        self.assertEqual(on, off)

    def test_echoed_prompt_is_the_same_text_and_specials(self):
        r = self.rig([THINK_NONCANON, SECOND, SECOND])
        prompts = r.converse(["hi", "more", "third"])
        off = self.rig([THINK_NONCANON, SECOND, SECOND], echo=False).converse(["hi", "more", "third"])
        for a, b in zip(prompts, off):
            self.assertEqual(r.tok.decode(a, errors="strict").encode(), r.tok.decode(b, errors="strict").encode())
            self.assertEqual([t for t in a if t >= 256 and t != AB], [t for t in b if t >= 256 and t != AB])

    def test_echo_keeps_carrying_earlier_replies(self):
        # turn 3 still carries turn 1's generated ids (a re-tokenized turn 1 would send the engine back to it)
        r = self.rig([THINK_NONCANON, SECOND, SECOND])
        p1, p2, p3 = r.converse(["hi", "more", "third"])
        self.assertEqual(p3[:len(p2) + len(SECOND) + 1], p2 + SECOND + r.end)
        self.assertEqual(p3[:len(p1) + len(THINK_NONCANON) + 1], p1 + THINK_NONCANON + r.end)
        self.assertEqual(r.svc.prompt_echo.summary()["entries"], 1)          # each turn replaced the one before

    def test_reply_in_another_format_falls_back_to_the_turn_before(self):
        # an empty thinking block comes back as "<think>\n\n</think>": the text differs inside the reply, so only
        # the turns before it are echoed - the prompt still starts with the previous one, as without the echo
        r = self.rig([list(b"</think>\n\nhello ") + [97, 98], SECOND])
        p1, p2 = r.converse(["hi", "more"])
        self.assertEqual(p2[:len(p1)], p1)
        self.assertEqual(r.svc.prompt_echo.stats["partial"], 1)

    def test_not_kept(self):
        cases = {
            "length": ([THINK_NONCANON], {"max_tokens": 5}),
            "ended on another stop token": ([THINK_NONCANON + [258]], {}),        # <|endoftext|>
            "special token written as text": ([list(b"x\n</think>\n\nsee <|im_start|> here")], {}),
            "stop string": ([THINK_NONCANON], {"stop": ["world"]}),
        }
        for why, (scripts, kw) in cases.items():
            with self.subTest(why=why):
                r = self.rig(scripts)
                r.chat([{"role": "user", "content": "hi"}], **kw)
                s = r.svc.prompt_echo.summary()
                self.assertEqual(s["recorded"], 0, s)
                self.assertEqual(s["refused"].get(why), 1, s)

    def test_count_tokens_is_not_echoed(self):
        r = self.rig([THINK_NONCANON, SECOND])
        r.converse(["hi"])
        msgs = [{"role": "user", "content": "hi"}]
        ids = r.svc.encode_prompt(msgs, None, {})
        self.assertNotIsInstance(ids, PromptIds)
        self.assertEqual(r.svc.prompt_echo.stats["full"] + r.svc.prompt_echo.stats["partial"], 0)


class Setup(unittest.TestCase):
    def svc(self, **attrs):
        tok = ByteTokenizer()
        s = Service(MockEngine(tok, "x"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        for k, v in attrs.items():
            setattr(s, k, v)
        return s

    def test_off_unless_asked(self):
        s = self.svc()
        with mock.patch.object(StrataEngine, "PARK_DIR", ""):
            prompt_echo_setup(s, {}, "")
        self.assertIsNone(s.prompt_echo)

    def test_env_overrides_the_config(self):
        with mock.patch.object(StrataEngine, "PARK_DIR", ""):
            s = self.svc()
            prompt_echo_setup(s, {"prompt_echo": True}, "0")
            self.assertIsNone(s.prompt_echo)
            s = self.svc()
            prompt_echo_setup(s, {}, "1")
            self.assertIsNotNone(s.prompt_echo)

    def test_off_where_kept_replies_could_be_lost(self):
        with mock.patch.object(StrataEngine, "PARK_DIR", "C:\\parked"):
            s = self.svc()
            self.assertIn("STRATA_PARK_DIR", prompt_echo_setup(s, {"prompt_echo": True}, ""))
            self.assertIsNone(s.prompt_echo)
        with mock.patch.object(StrataEngine, "PARK_DIR", ""):
            for attrs, word in (({"slot_save_path": "C:\\slots"}, "slot"), ({"vision": object()}, "images")):
                s = self.svc(**attrs)
                self.assertIn(word, prompt_echo_setup(s, {"prompt_echo": True}, ""))
                self.assertIsNone(s.prompt_echo)

    def test_kept_conversations_follow_the_engine_slots(self):
        with mock.patch.object(StrataEngine, "PARK_DIR", ""):
            s = self.svc()
            prompt_echo_setup(s, {"prompt_echo": True, "args": ["--conversation-cache-slots", "40"]}, "")
            self.assertEqual(s.prompt_echo.max_entries, 82)
            s = self.svc()
            prompt_echo_setup(s, {"prompt_echo": True}, "")
            self.assertEqual(s.prompt_echo.max_entries, 32)

    def test_bad_config_value(self):
        with self.assertRaises(ValueError):
            prompt_echo_setup(self.svc(), {"prompt_echo": "yes"}, "")


class Store(unittest.TestCase):
    END = 257

    def entry(self, *turns):
        """text, ids for turns of (text, ids) each ending with <|im_end|>."""
        text = "".join(t + "<|im_end|>" for t, _ in turns)
        ids = [x for _, i in turns for x in i + [self.END]]
        return text, ids

    def test_cut_at_the_last_agreeing_im_end(self):
        e = PromptEcho(self.END)
        text, ids = self.entry(("aa", [1, 2]), ("bb", [3]), ("cc", [4, 5]))
        self.assertTrue(e.record(text, ids, gen_at=len("aa<|im_end|>bb<|im_end|>")))
        hit = e.lookup("aa<|im_end|>bb<|im_end|>cX<|im_end|>more")
        self.assertEqual((hit.chars, hit.ids, hit.full), (len("aa<|im_end|>bb<|im_end|>"), [1, 2, 257, 3, 257], False))
        self.assertEqual(hit.differs_in_reply, 1)
        hit = e.lookup(text + "more")
        self.assertEqual((hit.chars, hit.ids, hit.full), (len(text), ids, True))
        self.assertEqual(e.lookup("zz").ids, [])

    def test_unpaired_im_end_is_refused(self):
        e = PromptEcho(self.END)
        self.assertFalse(e.record("a<|im_end|>b<|im_end|>", [1, self.END, 2, 3], gen_at=1))
        self.assertEqual(e.refused, {"unpaired <|im_end|>": 1})

    def test_least_recently_used_dropped_first(self):
        msgs = []
        e = PromptEcho(self.END, max_entries=2, log=msgs.append)
        for k in "xyz":
            if k == "z":
                e.lookup("x<|im_end|>")                      # x used again: y is the least recently used
            e.record(k + "<|im_end|>", [ord(k), self.END], gen_at=0)
        self.assertEqual([x.text[0] for x in e.entries], ["x", "z"])
        self.assertEqual(len(msgs), 1)

    def test_source_and_prefix_entries_are_replaced(self):
        e = PromptEcho(self.END)
        e.record("a<|im_end|>", [1, self.END], gen_at=0)
        e.record("q<|im_end|>", [9, self.END], gen_at=0)
        src = e.lookup("a<|im_end|>b").entry
        e.record("a<|im_end|>b<|im_end|>", [1, self.END, 2, self.END], gen_at=0, source=src)
        self.assertEqual(sorted(x.text for x in e.entries), ["a<|im_end|>b<|im_end|>", "q<|im_end|>"])

    def test_common_prefix_len(self):
        rnd = random.Random(7)
        for _ in range(300):
            n = rnd.choice([0, 1, 5, 40000, 70000])
            a = "".join(rnd.choice("ab") for _ in range(n))
            b = a[:rnd.randint(0, n)] + "".join(rnd.choice("ab") for _ in range(rnd.randint(0, 3)))
            self.assertEqual(common_prefix_len(a, b), len(os.path.commonprefix([a, b])))


if __name__ == "__main__":
    unittest.main()
