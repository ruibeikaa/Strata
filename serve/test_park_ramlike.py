"""serve/test_park_ramlike.py - PH402 local: the prompt echo's entries kept over a restart, and the RAM-like park
keeper (serve/park_keeper.py) holding named sections after the process that made them has gone.  No GPU, no engine.

    python -m unittest serve.test_park_ramlike -v
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import park_keeper  # noqa: E402
from serve.prompt_echo import PromptEcho  # noqa: E402
from serve.server import StrataEngine  # noqa: E402

END = 257


def entry(*turns):
    text = "".join(t + "<|im_end|>" for t, _ in turns)
    ids = [x for _, i in turns for x in i + [END]]
    return text, ids


class EchoExport(unittest.TestCase):
    def test_round_trip(self):
        a = PromptEcho(END)
        text, ids = entry(("aa", [1, 2]), ("bb", [3, 4, 5]))
        self.assertTrue(a.record(text, ids, gen_at=len("aa<|im_end|>")))
        a.record(*entry(("q", [9])), gen_at=0)
        b = PromptEcho(END)
        self.assertEqual(b.import_blob(a.export()), (2, ""))
        hit = b.lookup(text + "next")
        self.assertEqual((hit.ids, hit.full), (ids, True))
        self.assertEqual([e.text for e in b.entries], [e.text for e in a.entries][::-1])   # lookup made aa most recent

    def test_refusals(self):
        a = PromptEcho(END)
        a.record(*entry(("aa", [1])), gen_at=0)
        blob = a.export(now=1000.0)
        self.assertEqual(PromptEcho(END + 1).import_blob(blob)[0], 0)                       # another tokenizer
        n, why = PromptEcho(END).import_blob(blob, max_age_s=3600, now=1000.0 + 7200)
        self.assertEqual(n, 0)
        self.assertIn("ago", why)
        self.assertEqual(PromptEcho(END).import_blob(b"junk")[0], 0)

    def test_imported_go_before_the_current_and_respect_the_cap(self):
        a = PromptEcho(END)
        for k in "xyz":
            a.record(*entry((k, [ord(k)])), gen_at=0)
        b = PromptEcho(END, max_entries=3)
        b.record(*entry(("w", [119])), gen_at=0)
        self.assertEqual(b.import_blob(a.export())[0], 3)
        self.assertEqual([e.text[0] for e in b.entries], ["y", "z", "w"])   # x (the oldest) dropped by the cap

    def test_engine_takes_entries_loaded_before_the_echo_exists(self):
        a = PromptEcho(END)
        a.record(*entry(("aa", [1])), gen_at=0)
        eng = StrataEngine.__new__(StrataEngine)
        eng.echo_import(a.export())
        self.assertIsNotNone(eng._echo_pending)
        eng.echo = PromptEcho(END)
        pending, eng._echo_pending = eng._echo_pending, None
        eng.echo_import(pending)
        self.assertEqual(len(eng.echo.entries), 1)


MAKER = r"""
import sys, time
sys.path.insert(0, sys.argv[1])
from serve import park_keeper as K
names = sys.argv[3:]
held = []
for i, n in enumerate(names):
    s = K.Section(n, 3 << 20)
    s.write(bytes([i + 1]) * (3 << 20))
    held.append(s)
print("MADE", flush=True)
sys.stdin.readline()          # the test starts the keeper meanwhile; then this process exits, unmapping its views
"""


@unittest.skipUnless(os.name == "nt", "named sections are Windows only")
class Keeper(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="strata-park-ramlike-test-")
        self.stamp = f"{os.getpid()}-{time.time_ns()}"
        self.convs = [f"ctest{self.stamp}-{k}" for k in ("a", "b")]
        with open(os.path.join(self.dir, park_keeper.MANIFEST), "w", encoding="ascii") as f:
            f.write("strata-park 2\ntag 0123456789abcdef\nstages 2\n")
            for c in self.convs:
                f.write(f"conv {c} 60000 {int(time.time())}\n")
            f.write("end\n")

    def tearDown(self):
        park_keeper.release(self.dir)

    def test_manifest_names(self):
        names, head = park_keeper.manifest_sections(self.dir)
        self.assertEqual(head, {"tag": "0123456789abcdef", "stages": 2})
        self.assertEqual(names, [f"Local\\strata-park-{c}-s{s}" for c in self.convs for s in (0, 1)])

    def test_sections_outlive_their_maker(self):
        names, _ = park_keeper.manifest_sections(self.dir)
        root = str(Path(__file__).resolve().parents[1])
        maker = subprocess.Popen([sys.executable, "-c", MAKER, root, "-"] + names, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, text=True)
        self.assertEqual(maker.stdout.readline().strip(), "MADE")
        echo_sec, echo_name = park_keeper.put_echo(b"echo-bytes")
        doc = park_keeper.spawn(self.dir, 600, echo_name, timeout_s=60)
        echo_sec.close()
        maker.stdin.write("\n")
        maker.stdin.flush()
        maker.wait(30)
        self.assertEqual((doc["sections"], doc["missing"], doc["echo"]), (5, [], echo_name))
        self.assertGreaterEqual(doc["bytes"], 4 * (3 << 20))
        # the maker is gone; the keeper still holds every section and the echo
        for i, n in enumerate(names):
            s = park_keeper.Section(n)
            self.assertEqual(s.read(), bytes([i + 1]) * (3 << 20))
            s.close()
        self.assertEqual(park_keeper.get_echo(echo_name), b"echo-bytes")
        self.assertIsNotNone(park_keeper.status(self.dir))
        # the keeper is no one's child: its parent (the middle process) has exited
        park_keeper.release(self.dir)
        end = time.monotonic() + 15
        while time.monotonic() < end and park_keeper.pid_alive(doc["pid"]):
            time.sleep(0.2)
        self.assertFalse(park_keeper.pid_alive(doc["pid"]), "released: the keeper ends")
        with self.assertRaises(OSError):
            park_keeper.Section(names[0])                  # gone with the last handle

    def test_newer_manifest_ends_the_keeper(self):
        names, _ = park_keeper.manifest_sections(self.dir)
        made = [park_keeper.Section(n, 1 << 20) for n in names]
        doc = park_keeper.spawn(self.dir, 600, timeout_s=60)
        for s in made:
            s.close()
        time.sleep(0.05)
        os.utime(os.path.join(self.dir, park_keeper.MANIFEST), ns=(time.time_ns(), time.time_ns() + 10**9))
        end = time.monotonic() + 15
        while time.monotonic() < end and park_keeper.pid_alive(doc["pid"]):
            time.sleep(0.2)
        self.assertFalse(park_keeper.pid_alive(doc["pid"]))

    def test_spawn_gives_up_within_its_whole_wait(self):
        """2026-10-30 16:00: a keeper that is late must not hold the unload past the stop script's 120 s POST timeout."""
        names, _ = park_keeper.manifest_sections(self.dir)
        made = [park_keeper.Section(n, 1 << 20) for n in names]
        t0 = time.monotonic()
        with self.assertRaises(RuntimeError) as cm:
            park_keeper.spawn(self.dir, 600, timeout_s=4, python=os.environ.get("COMSPEC", "cmd.exe"))   # never a keeper
        for s in made:
            s.close()
        self.assertLess(time.monotonic() - t0, 12)
        self.assertIn("no keeper ready", str(cm.exception))
        self.assertIn("keeper.json", str(cm.exception))                 # and says what the folder shows

    def test_touch_stops_at_its_deadline_and_goes_on_later(self):
        s = park_keeper.Section(f"Local\strata-park-touch-{self.stamp}", 3 << 20)
        s.write(b"x" * (3 << 20))
        seen = []
        with mock.patch.object(park_keeper, "TOUCH_CHUNK", 1 << 20),                 mock.patch.object(s, "_touch_range", lambda a, b: seen.append((a, b))):
            self.assertFalse(s.touch(until=time.monotonic() - 1))       # no time: nothing touched, nowhere to resume
            self.assertEqual(seen, [])
            calls = {"n": 0}
            real = time.monotonic

            def clock():                                                # time runs out after the second step
                calls["n"] += 1
                return real() + (100 if calls["n"] > 2 else 0)
            with mock.patch.object(park_keeper.time, "monotonic", clock):
                self.assertFalse(s.touch(until=real() + 50))
            self.assertEqual(len(seen), 2)
            self.assertTrue(s.touch())                                  # the rest from where it stopped
            self.assertEqual(seen[2][0], seen[1][1])
            self.assertEqual(seen[-1][1], 64 + (3 << 20))               # up to the end of the data (head included)
        s.close()

    def test_no_keeper_no_status(self):
        self.assertIsNone(park_keeper.status(self.dir))


if __name__ == "__main__":
    unittest.main()
