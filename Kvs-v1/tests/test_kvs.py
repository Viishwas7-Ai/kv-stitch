"""Every cached path must give exactly the plan the whole prompt gives."""
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
from kvs import KVPlanner, parse_workflows  # noqa: E402

MODEL = os.environ.get("KVS_MODEL", os.path.join(HERE, "..", "..", "tiny", "tiny.gguf"))
pytestmark = pytest.mark.skipif(not os.path.exists(MODEL), reason="no model")

HEADER = "You turn a user command into a json plan of actions.\n"
GLUE = ("glue", "Use only the actions below.\n")
CREATE = ("create", "CreateFolder makes a folder. CreateFile makes a file.\n", True)
NOTES = ("notes", "NotesAction creates or reads a note by title.\n", True)
ZIP = ("zip", "ZipAction zips a folder.\n", True)
RULES = ("rules", "Rules: answer with json only.\n")
TAIL = "User command: make a folder and a note\nPlan:"


@pytest.fixture()
def kp(tmp_path):
    p = KVPlanner({"tiny": MODEL}, str(tmp_path), n_ctx=2048, n_gpu_layers=0, chat=False, max_tokens=12)
    yield p
    p.close()


def reference(kp, pieces):
    """No cache: the whole prompt read from scratch."""
    kp.load("tiny")
    toks = kp.eng.tok(HEADER, bos=True) + [t for p in pieces for t in kp.eng.tok(p[1])] + kp.eng.tok(TAIL)
    kp.eng.clear()
    logits = kp.eng.decode(toks, 0, want_last=True)
    return kp.eng.generate(logits, len(toks), 12)[0]


def test_miss_then_full(kp):
    pieces = [GLUE, CREATE, NOTES, RULES]
    ref = reference(kp, pieces)
    a = kp.plan("tiny", HEADER, pieces, TAIL)
    b = kp.plan("tiny", HEADER, pieces, TAIL)
    assert (a.path, b.path) == ("miss", "full") and a.text == b.text == ref


def test_first_run_start_gives_partial(kp):
    built = kp.build_starts("tiny", HEADER, [[GLUE, CREATE], [GLUE, NOTES], [GLUE, ZIP]])
    assert built == 3
    assert kp.build_starts("tiny", HEADER, [[GLUE, CREATE]]) == 0          # unchanged: skipped
    pieces = [GLUE, ZIP, CREATE, RULES]
    r = kp.plan("tiny", HEADER, pieces, TAIL)
    assert r.path == "partial" and r.detail["cached_pieces"] == 3 and r.text == reference(kp, pieces)


def test_checkpoint_reused_by_another_mix(kp):
    kp.plan("tiny", HEADER, [GLUE, NOTES, CREATE, RULES], TAIL)        # saves start up to NOTES
    r = kp.plan("tiny", HEADER, [GLUE, NOTES, ZIP, RULES], TAIL)
    assert r.path == "partial"


def test_changed_text_rebuilds_and_removes_old(kp):
    kp.build("tiny", HEADER, [GLUE, CREATE])
    old = dict(kp.cache.index)
    kp.build("tiny", HEADER, [GLUE, ("create", "CreateFolder makes a NEW folder.\n", True)])
    assert len(kp.cache.index) == 1 and set(kp.cache.index) != set(old)
    k = next(iter(kp.cache.index))
    assert kp.cache.index[k]["pinned"]
    assert "NEW folder" in open(os.path.join(kp.cache.dir, k + ".prompt.txt")).read()
    assert not os.path.exists(os.path.join(kp.cache.dir, next(iter(old)) + ".kv"))


def test_one_folder_per_model(kp, tmp_path):
    kp.models["tiny2"] = MODEL
    kp.build("tiny", HEADER, [GLUE, CREATE])
    kp.build("tiny2", HEADER, [GLUE, CREATE])
    assert sorted(os.listdir(tmp_path)) == ["tiny", "tiny2"]


def test_workflows(kp):
    wfs = parse_workflows("a: create, notes\nb: zip")
    def build(mods):
        m = {"create": CREATE, "notes": NOTES, "zip": ZIP}
        return HEADER, [GLUE] + [m[x] for x in mods] + [RULES]
    rep = kp.add_workflows("tiny", wfs, build)
    assert rep["workflows"] == 2 and rep["built"] == 4
    pieces = build(["create", "notes"])[1]
    r = kp.plan("tiny", HEADER, pieces, TAIL)
    assert r.path == "full" and r.text == reference(kp, pieces)
    r = kp.plan("tiny", HEADER, build(["create", "notes", "zip"])[1], TAIL)
    assert r.path == "partial"


def test_fallback(tmp_path):
    kp = KVPlanner({"x": "/does/not/exist.gguf"}, str(tmp_path), fallback=lambda m, p: '{"plan": []}')
    r = kp.plan("x", HEADER, [GLUE, CREATE], TAIL)
    assert r.path == "fallback" and r.text == '{"plan": []}'


def test_request_variant_never_deletes_a_pinned_start(kp):
    kp.build("tiny", HEADER, [GLUE, CREATE, RULES])                     # first run, pinned
    pinned = set(kp.cache.index)
    kp.plan("tiny", HEADER, [GLUE, CREATE, ("rules", "Rules: json only, today.\n")], TAIL)
    assert pinned <= set(kp.cache.index)                                # still there


def test_engine_stitch_with_whole_module_is_exact(kp):
    """Mechanics: stitching module cells computed after the SAME text is exactly reading them."""
    kp.load("tiny")
    e = kp.eng
    a, m = e.tok(HEADER, bos=True), e.tok(NOTES[1])
    e.clear(); e.decode(a + m, 0); blob = e.save()
    e.clear(); e.decode(a, 0)
    e.stitch(blob, len(a), len(a) + len(m), len(a))
    t = e.tok(TAIL)
    out = e.generate(e.decode(t, len(a) + len(m), want_last=True), len(a) + len(m) + len(t), 12)[0]
    e.clear()
    full = e.generate(e.decode(a + m + t, 0, want_last=True), len(a + m + t), 12)[0]
    assert out == full


def test_speed_mode_stitches_every_module_after_the_start(kp):
    base = [GLUE]
    kp.build_starts("tiny", HEADER, [base + [CREATE], base + [NOTES], base + [ZIP]])
    before = len(kp.cache.index)
    r = kp.plan("tiny", HEADER, [GLUE, CREATE, NOTES, ZIP, RULES], TAIL, fast=True)
    assert r.path == "fast" and r.detail["stitched"] == 2 and r.detail["cached_pieces"] == 3
    assert len(kp.cache.index) == before            # used the first-run starts, built nothing new


def test_speed_mode_uses_exact_when_whole_prompt_cached(kp):
    pieces = [GLUE, CREATE, NOTES, RULES]
    kp.plan("tiny", HEADER, pieces, TAIL)
    assert kp.plan("tiny", HEADER, pieces, TAIL, fast=True).path == "full"
    assert kp.plan("tiny", HEADER, [GLUE, CREATE, RULES], TAIL, fast=True).path != "fast"   # 1 module
