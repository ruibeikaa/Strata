"""serve/test_reasoning_close_retry.py - #1053 reasoning_close_retry and the two guards added on top of it.

A reply that ends on its stop token still inside <think>, with no answer and no call, is continued once with the
thinking closed (opt-in, the config's "reasoning_close_retry": true).  On top of upstream's trigger:
 - not while a call written inside the thinking waits (OutputParser.pending, #804/#1058): the stop already makes it a
   call, and a second pass in which the model wrote it again would deliver it twice;
 - also when the reply stopped inside a `<tool_call>` held in the thinking (parser state "rcall", seen live as
   '<tool_call>\\n<|im_start|>assistant'): the held text goes out as reasoning, then </think>, then the continuation.
A reply that still has no answer leaves one line in the server log, whether the retry is on or not.

The prompt echo (serve/prompt_echo.py) never keeps a reply the server wrote tokens into, so the forced </think> is
never carried as generated ids: the next turn is the earlier turns' ids plus the re-tokenized text, as without the echo.

Mock engine, no GPU, no pack.

    python -m unittest serve.test_reasoning_close_retry -v
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import threading
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, OutputParser  # noqa: E402
from serve.prompt_echo import common_prefix_len  # noqa: E402
from serve.server import REASONING_CLOSE, ByteTokenizer, MockEngine, Service, serve  # noqa: E402
from serve.test_prompt_echo import AB, Rig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CALL = "<tool_call>\n<function=read>\n<parameter=path>\nx\n</parameter>\n</function>\n</tool_call>"
TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}}}}]
SCHEMA = [t["function"] for t in TOOLS]
NO_ANSWER = "with no answer (no content, no tool call"          # the log line of a reply left without one
RETRIED = "closing the thinking once and continuing (reasoning_close_retry)"

# What the model writes before its stop token, and after the forced </think> (the model writing the call again in A)
A_CALL = ("Plan: read x.\n" + CALL + "\n", CALL)
B_ANSWER = ("Plan. Let me write.\n## 答案\n| a |\n", "## 答案\n| a |")
C_HELD = ("Plan.\nLet me do it.\n<tool_call>\n<|im_start|>assistant", CALL)
D_HELD_AFTER_CALL = ("Plan.\n" + CALL + "\n<tool_call>\n<function=read>\n<parameter=path>\ny", CALL)


class PassEngine(MockEngine):
    """One script per engine pass (the last repeats); every prompt it is given is kept."""

    def __init__(self, tok, scripts, cancel_at_stop=None):
        super().__init__(tok, list(scripts))
        self.prompts, self.cancel_at_stop = [], cancel_at_stop

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        for t in super().generate(ids, max_new, sampling, cancel, embeddings):
            if self.cancel_at_stop is not None and t == self.scripts[0][-1]:
                self.cancel_at_stop.set()           # the client cancels just as the stop token arrives
            yield t


class Over(unittest.TestCase):
    """The shapes over the OpenAI API, with the retry off and on."""

    def rig(self, scripts, retry):
        tok = ByteTokenizer()
        engine = PassEngine(tok, scripts)
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.reasoning_close_retry = retry
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return tok, engine, f"http://127.0.0.1:{httpd.server_address[1]}"

    def chat(self, scripts, retry, stream=False, **extra):
        tok, engine, base = self.rig(scripts, retry)
        body = {"model": "m", "messages": [{"role": "user", "content": "go"}], "tools": TOOLS, "max_tokens": 400,
                "stream": stream, **extra}
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        out = io.StringIO()
        with contextlib.redirect_stdout(out), urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
        if stream:
            return tok, engine, [json.loads(x[6:]) for x in raw.splitlines() if x.startswith("data: {")], out.getvalue()
        return tok, engine, json.loads(raw)["choices"][0], out.getvalue()

    def test_a_call_inside_the_thinking_is_one_call_in_one_pass(self):
        # the guard: the stop already makes the waiting call a call; no second pass that could write it again
        for retry in (False, True):
            with self.subTest(retry=retry):
                _, engine, c, log = self.chat(A_CALL, retry)
                self.assertEqual(len(engine.prompts), 1)
                self.assertEqual([t["function"]["name"] for t in c["message"]["tool_calls"]], ["read"])
                self.assertEqual(json.loads(c["message"]["tool_calls"][0]["function"]["arguments"]), {"path": "x"})
                self.assertFalse(c["message"].get("content"))
                self.assertEqual(c["finish_reason"], "tool_calls")
                self.assertNotIn(RETRIED, log)
                self.assertNotIn(NO_ANSWER, log)

    def test_an_answer_inside_the_thinking_comes_out_as_content(self):
        tok, engine, c, log = self.chat(B_ANSWER, True)
        self.assertEqual(len(engine.prompts), 2)
        first, second = engine.prompts
        # the continuation is this pass's prompt + what it wrote (not its stop token) + the closing: a prefix + more
        self.assertEqual(second, first + tok.encode(B_ANSWER[0]) + tok.encode(REASONING_CLOSE, parse_special=True))
        self.assertEqual(c["message"]["content"], B_ANSWER[1])
        self.assertIn("## 答案", c["message"]["reasoning_content"])
        self.assertEqual(c["finish_reason"], "stop")
        self.assertEqual(log.count(RETRIED), 1)
        self.assertNotIn(NO_ANSWER, log)

    def test_off_the_answer_stays_in_the_thinking_and_is_logged(self):
        _, engine, c, log = self.chat(B_ANSWER, False)
        self.assertEqual(len(engine.prompts), 1)
        self.assertFalse(c["message"].get("content"))
        self.assertIn("## 答案", c["message"]["reasoning_content"])
        self.assertEqual(log.count(NO_ANSWER), 1)
        self.assertIn("reasoning_close_retry: true in strata-<model>.json", log)

    def test_a_stop_inside_a_held_call_is_closed_and_continued(self):
        _, engine, c, log = self.chat(C_HELD, True)
        self.assertEqual(len(engine.prompts), 2)
        self.assertIn("<tool_call>\n<|im_start|>assistant", c["message"]["reasoning_content"])
        self.assertEqual([t["function"]["name"] for t in c["message"]["tool_calls"]], ["read"])
        self.assertFalse(c["message"].get("content"))
        self.assertEqual(c["finish_reason"], "tool_calls")
        self.assertEqual(log.count(RETRIED), 1)
        self.assertNotIn(NO_ANSWER, log)

    def test_off_a_stop_inside_a_held_call_is_reasoning_text_and_logged(self):
        _, engine, c, log = self.chat(C_HELD, False)
        self.assertEqual(len(engine.prompts), 1)
        self.assertIn("<tool_call>\n<|im_start|>assistant", c["message"]["reasoning_content"])
        self.assertFalse(c["message"].get("tool_calls"))
        self.assertFalse(c["message"].get("content"))
        self.assertEqual(log.count(NO_ANSWER), 1)

    def test_a_held_call_after_a_waiting_call_is_not_retried(self):
        # a waiting call keeps the retry off even when the stop came inside the next held call: logged instead
        _, engine, c, log = self.chat(D_HELD_AFTER_CALL, True)
        self.assertEqual(len(engine.prompts), 1)
        self.assertNotIn(RETRIED, log)
        self.assertFalse(c["message"].get("content"))
        self.assertEqual(log.count(NO_ANSWER), 1)
        self.assertIn("reasoning_close_retry did not apply", log)

    def test_streamed_the_held_text_is_reasoning_before_the_call(self):
        _, engine, chunks, _ = self.chat(C_HELD, True, stream=True)
        self.assertEqual(len(engine.prompts), 2)
        order = []
        for ch in chunks:
            for d in [c["delta"] for c in ch.get("choices") or []]:
                if d.get("reasoning_content"):
                    order.append(("reasoning", d["reasoning_content"]))
                if d.get("content"):
                    order.append(("content", d["content"]))
                if d.get("tool_calls"):
                    order.append(("tool", ""))
        reasoning = "".join(t for k, t in order if k == "reasoning")
        self.assertIn("<tool_call>\n<|im_start|>assistant", reasoning)
        first_tool = [k for k, _ in order].index("tool")
        self.assertTrue(all(k == "reasoning" for k, _ in order[:first_tool]))
        self.assertFalse([k for k, _ in order[first_tool:] if k == "reasoning"])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_a_stop_string_cuts_the_answer_after_the_close(self):
        # stop strings act on the answer only: the thinking holds "答案" too, and the retry still happens
        _, engine, c, _ = self.chat(B_ANSWER, True, stop=["答案"])
        self.assertEqual(len(engine.prompts), 2)
        self.assertEqual((c["message"]["content"], c["finish_reason"]), ("## ", "stop"))


class Paths(unittest.TestCase):
    """The retry's own conditions, through Service.run: length, no room, cancel, once only, no thinking."""

    def run_(self, scripts, max_new=400, thinking=True, retry=True, cancel_at_stop=False):
        tok = ByteTokenizer()
        cancel = threading.Event()
        engine = PassEngine(tok, scripts, cancel if cancel_at_stop else None)
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.reasoning_close_retry = retry
        ids = tok.encode("<|im_start|>user\ngo<|im_end|>\n<|im_start|>assistant\n<think>\n", parse_special=True)
        evs, done, out = [], None, io.StringIO()
        with contextlib.redirect_stdout(out):
            for kind, x in svc.run(ids, thinking, SCHEMA, max_new, {}, cancel):
                if kind == "event":
                    evs.append(x)
                elif kind == "done":
                    done = x
        return engine, evs, done, out.getvalue()

    def test_a_length_cut_inside_the_thinking_is_not_retried(self):
        engine, evs, done, log = self.run_(B_ANSWER, max_new=10)
        self.assertEqual((len(engine.prompts), done["finish"]), (1, "length"))
        self.assertIn("reached max tokens while still thinking", log)       # #530, as before
        self.assertNotIn(NO_ANSWER, log)

    def test_no_room_left_to_close(self):
        n = len(ByteTokenizer().encode(B_ANSWER[0])) + 1                 # the thought and its stop token, nothing more
        engine, evs, done, log = self.run_(B_ANSWER, max_new=n + 2)
        self.assertEqual((len(engine.prompts), done["finish"]), (1, "stop"))
        self.assertFalse([e for e in evs if e.kind == "content"])
        self.assertIn("reasoning_close_retry did not apply", log)

    def test_a_cancel_at_the_stop_is_not_retried(self):
        engine, evs, done, log = self.run_(B_ANSWER, cancel_at_stop=True)
        self.assertEqual((len(engine.prompts), done["finish"]), (1, "cancel"))
        self.assertNotIn(NO_ANSWER, log)

    def test_closed_once_only(self):
        # the continuation writes nothing: no third pass, and the log says the retry brought no answer either
        engine, evs, done, log = self.run_((B_ANSWER[0], ""))
        self.assertEqual((len(engine.prompts), done["finish"]), (2, "stop"))
        self.assertFalse([e for e in evs if e.kind == "content"])
        self.assertIn("brought none either", log)

    def test_without_thinking_there_is_nothing_to_close(self):
        engine, evs, done, log = self.run_(("just text", "x"), thinking=False)
        self.assertEqual(len(engine.prompts), 1)
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "just text")
        self.assertNotIn(NO_ANSWER, log)

    def test_a_reply_that_answered_is_not_touched(self):
        engine, evs, done, log = self.run_(("ok</think>\n\nFine.", "x"))
        self.assertEqual(len(engine.prompts), 1)
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "Fine.")
        self.assertNotIn(NO_ANSWER, log)


class Parser(unittest.TestCase):
    """What the forced closing does to the parser in each state the retry now takes."""

    def test_rcall_the_held_text_goes_out_as_reasoning_before_the_close(self):
        p = OutputParser(thinking=True, tools=SCHEMA, stream_tools=True)
        evs = p.feed(C_HELD[0])
        self.assertEqual((p.state, p.pending), ("rcall", []))
        evs = p.feed(REASONING_CLOSE)
        self.assertEqual([e.kind for e in evs], ["reasoning"])
        self.assertEqual(evs[0].text, "<tool_call>\n<|im_start|>assistant\n")
        self.assertEqual(p.state, "content")
        evs = p.feed(CALL) + p.finish("stop")
        self.assertEqual([e.call.name for e in evs if e.kind == "tool_call"], ["read"])
        self.assertEqual((p.rescued, p.refused), (0, 0))

    def test_a_waiting_call_is_what_the_close_would_have_released(self):
        # why the retry stays off: with a call waiting, the stop releases it - and so would the closing, before a
        # second pass that could write the call again
        p = OutputParser(thinking=True, tools=SCHEMA, stream_tools=True)
        p.feed(A_CALL[0])
        self.assertEqual((p.state, len(p.pending)), ("reasoning", 1))
        self.assertEqual([e.call.name for e in p.finish("stop") if e.kind == "tool_call"], ["read"])


class Echo(unittest.TestCase):
    """(a) the prompt echo after a closed reply; (b) what the next turn shares with what the engine holds."""
    # the thought holds a pair the BPE would merge ("ab" -> AB): a reply the echo would normally carry as generated
    THOUGHT = list(b"let me think: ") + [97, 98] + list(b" is it")
    ANSWER = list(b"hello world")
    THIRD = list(b"fine\n</think>\n\nthird")

    def rig(self, scripts, echo=True):
        r = Rig(scripts, echo)
        r.svc.reasoning_close_retry = True
        self.addCleanup(r.close)
        return r

    def converse(self, r, turns):
        with contextlib.redirect_stdout(io.StringIO()):
            return r.converse(turns)

    def test_a_closed_reply_is_not_kept_and_the_next_turn_is_retokenized(self):
        r = self.rig([self.THOUGHT, self.ANSWER, self.THIRD, self.THIRD])
        p1, p1b, p2, p3 = self.converse(r, ["hi", "more", "again"])
        close = r.tok.encode(REASONING_CLOSE, parse_special=True)
        self.assertEqual(p1b, p1 + self.THOUGHT + close)                       # the retry's continuation
        s = r.svc.prompt_echo.summary()
        self.assertEqual(s["refused"], {"server-written tokens": 1})
        self.assertEqual(s["recorded"], 2)                                      # turns 2 and 3 were kept
        # turn 2 carries no generated id of the closed reply: "ab" comes back as the BPE's AB, the closing as text
        reply = p2[len(p1):]
        self.assertNotIn([97, 98], [reply[i:i + 2] for i in range(len(reply) - 1)])
        self.assertIn(AB, reply)
        off = self.rig([self.THOUGHT, self.ANSWER, self.THIRD, self.THIRD], echo=False)
        q1, q1b, q2, q3 = self.converse(off, ["hi", "more", "again"])
        self.assertEqual(p2, q2)                                                # exactly the echo-off ids
        # turn 3 is echoed again from turn 2's reply (the echo works on after a closed reply)
        self.assertEqual(r.svc.prompt_echo.stats["full"], 1)
        self.assertEqual(p3[:len(p2)], p2)

    def test_the_next_turn_leaves_the_engine_inside_the_closed_reply_only(self):
        # (b): what the engine holds after the closed reply is the continuation's prompt + its answer + <|im_end|>;
        # the next prompt shares all of the turn before (nothing earlier is read again) ...
        r = self.rig([self.THOUGHT, self.ANSWER, self.THIRD])
        p1, p1b, p2 = self.converse(r, ["hi", "more"])
        held = p1b + self.ANSWER + r.end
        shared = common_prefix_len(held, p2)
        self.assertGreaterEqual(shared, len(p1))
        self.assertEqual(shared, len(p1) + len(list(b"let me think: ")))     # ... up to the merged pair
        # with a thought the BPE writes the same way, the closed reply comes back id for id: nothing to read again
        canon = list(b"let me think: plus is it")
        r = self.rig([canon, self.ANSWER, self.THIRD])
        p1, p1b, p2 = self.converse(r, ["hi", "more"])
        held = p1b + self.ANSWER + r.end
        self.assertEqual(p2[:len(held)], held)


if __name__ == "__main__":
    unittest.main()
