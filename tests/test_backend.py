"""PlannerBackend: every cached path must equal the full prompt; fast mode only when allowed."""
import os

import pytest

from kvstitch import PlannerBackend

MODEL = os.environ.get("KVSTITCH_MODEL", "tiny/tiny.gguf")
pytestmark = pytest.mark.skipif(not os.path.exists(MODEL), reason="no model")

HEADER = "You turn a user command into a json plan of actions.\n"
GLUE = ("glue", "Use only the actions below.\n")
CREATE = ("create", "CreateFolder makes a folder. CreateFile makes a file.\n", True)
NOTES = ("notes", "NotesAction creates or reads a note by title.\n", True)
ZIP = ("zip", "ZipAction zips a folder.\n", True)
OS_ = ("os", "OS: macOS. Default path ~/Desktop.\n")
TAIL = "User command: make a folder and a note\nPlan:"


@pytest.fixture()
def be(tmp_path):
    b = PlannerBackend({"tiny": MODEL}, str(tmp_path), n_ctx=2048, n_gpu_layers=0, chat=False,
                       max_tokens=12, save_after=1, max_disk_mb=None)
    yield b
    b.close()


def full_text(b, pieces):
    b._load("tiny")
    names, _ = b._prepare(HEADER, pieces)
    out, _ = b._st.run_full(names, TAIL, 12)
    return out


def test_miss_then_full_is_exact(be):
    pieces = [GLUE, CREATE, NOTES, OS_]
    ref = full_text(be, pieces)
    r1 = be.plan("tiny", HEADER, pieces, TAIL)
    r2 = be.plan("tiny", HEADER, pieces, TAIL)
    assert (r1.path, r2.path) == ("miss", "full")
    assert r1.text == r2.text == ref


def test_shared_start_is_reused_exactly(be):
    be.warm("tiny", HEADER, [[GLUE, CREATE]])          # core: everything up to the first module
    pieces = [GLUE, CREATE, ZIP, OS_]
    ref = full_text(be, pieces)
    r = be.plan("tiny", HEADER, pieces, TAIL)
    assert r.path == "partial" and r.text == ref


def test_checkpoint_saved_on_a_miss(be):
    be.plan("tiny", HEADER, [GLUE, NOTES, CREATE, OS_], TAIL)   # saves "up to NOTES" on the way
    r = be.plan("tiny", HEADER, [GLUE, NOTES, ZIP, OS_], TAIL)
    assert r.path == "partial"


def test_fast_mode_only_when_allowed(be):
    pieces = [GLUE, CREATE, NOTES, OS_]
    assert be.plan("tiny", HEADER, pieces, TAIL, fast=True).path == "fast"
    be.fast_max_modules = 1
    assert be.plan("tiny", HEADER, [GLUE, ZIP, CREATE, OS_], TAIL, fast=True).path != "fast"
    be.plan("tiny", HEADER, pieces, TAIL)               # now cached
    assert be.plan("tiny", HEADER, pieces, TAIL, fast=True).path == "full"   # exact wins


def test_changed_text_is_not_reused(be):
    be.plan("tiny", HEADER, [GLUE, CREATE, OS_], TAIL)
    r = be.plan("tiny", HEADER, [GLUE, CREATE, ("os", "OS: macOS. Default path ~/Documents.\n")], TAIL)
    assert r.path != "full"


def test_fallback(tmp_path):
    seen = {}
    def fb(model, prompt):
        seen["prompt"] = prompt
        return '{"plan": []}'
    b = PlannerBackend({"tiny": "/does/not/exist.gguf"}, str(tmp_path), fallback=fb)
    r = b.plan("tiny", HEADER, [GLUE, CREATE], TAIL)
    assert r.path == "fallback" and r.text == '{"plan": []}' and TAIL in seen["prompt"]


def build_factory():
    table = {"CREATE": CREATE, "NOTES": NOTES, "ZIP": ZIP}
    def build(mods):
        return HEADER, [GLUE] + [table[m] for m in mods] + [OS_]
    return build


def test_parse_workflows():
    from kvstitch import parse_workflows
    wfs = parse_workflows("""
        # comment
        make_and_note: CREATE, NOTES
        zip_only: ZIP   @tiny   # trailing comment
    """)
    assert [(w.name, w.modules, w.model) for w in wfs] == [
        ("make_and_note", ["CREATE", "NOTES"], None), ("zip_only", ["ZIP"], "tiny")]
    with pytest.raises(ValueError):
        parse_workflows("bad line without colon")


def test_workflow_exact_and_with_extras(be):
    from kvstitch import parse_workflows
    build = build_factory()
    rep = be.add_workflows("tiny", parse_workflows("make_and_note: CREATE, NOTES\nother: ZIP @nope"), build)
    assert rep["workflows"] == 1 and rep["built"] == 2
    # exactly the workflow -> whole prefix cached
    h, p = build(["CREATE", "NOTES"])
    r = be.plan("tiny", h, p, TAIL)
    assert r.path == "full" and r.text == full_text(be, p)
    # workflow + an extra module after it -> its start is reused, exactly
    h, p = build(["CREATE", "NOTES", "ZIP"])
    r = be.plan("tiny", h, p, TAIL)
    assert r.path == "partial" and r.text == full_text(be, p)
    # pinned
    assert be.stats()["pinned"] >= 2


def test_changed_workflow_rebuilds_and_old_is_removed(be):
    from kvstitch import parse_workflows
    build = build_factory()
    r1 = be.add_workflows("tiny", parse_workflows("w: CREATE, NOTES"), build)
    r2 = be.add_workflows("tiny", parse_workflows("w: CREATE, NOTES"), build)
    assert r1["built"] == 2 and r2["built"] == 0 and r2["removed_old"] == 0   # unchanged: nothing to do
    # the module text changes (e.g. a module was edited): only that workflow is rebuilt
    def build2(mods):
        h, p = build(mods)
        return h, [(n, t + "Edited.\n" if n == "notes" else t, *rest) for n, t, *rest in p]
    r3 = be.add_workflows("tiny", parse_workflows("w: CREATE, NOTES"), build2)
    assert r3["built"] >= 1 and r3["removed_old"] >= 1
    assert be.stats()["pinned"] == 2
