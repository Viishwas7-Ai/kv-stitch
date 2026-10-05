"""Exactness tests for the cache plumbing (run with any GGUF transformer model).

    KVSTITCH_MODEL=path/to/model.gguf pytest -q tests
"""
import os

import pytest

from kvstitch import Stitcher
from kvstitch.core import MAIN

MODEL = os.environ.get("KVSTITCH_MODEL", "tiny/tiny.gguf")
pytestmark = pytest.mark.skipif(not os.path.exists(MODEL), reason="no model")

HEADER = "You turn a user command into a json plan of actions. Use only the actions below.\n"
MODS = {
    "CREATE": "CreateFolder makes a folder at a path. CreateFile makes a file. WriteToFile writes text.\n",
    "NOTES": "NotesAction creates, appends or reads an Apple note by title.\n",
    "ZIP": "ZipAction zips a folder or a list of files and can unzip an archive.\n",
}
TAIL = "User command: create a folder named demo on the desktop and zip it\nPlan:"


@pytest.fixture(scope="module")
def st():
    s = Stitcher(MODEL, n_ctx=2048, n_gpu_layers=0)
    s.set_header(HEADER)
    for k, v in MODS.items():
        s.add_module(k, v)
    yield s
    s.close()


def diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def full_logits(st, toks):
    st.clear()
    return st._decode(toks, 0, MAIN, want_last=True)


def test_single_module_is_exact(st):
    """header + one module: the module saw exactly this header, so nothing is approximated."""
    toks, pos = st.assemble(["CREATE"])
    tail = st.tok(TAIL)
    stitched = st._decode(tail, pos, MAIN, want_last=True)
    assert diff(stitched, full_logits(st, toks + tail)) < 1e-3


def test_shift_roundtrip(st):
    """Shift a module forward then back: must match the unshifted result (RoPE shift is applied)."""
    import llama_cpp as lc
    h = len(st.header.tokens)
    m = st.modules["NOTES"]
    n, d = len(m.tokens), 37
    tail = st.tok(TAIL)
    _, pos = st.assemble(["NOTES"])
    ref = st._decode(tail, pos, MAIN, want_last=True)
    _, pos = st.assemble(["NOTES"])
    lc.llama_memory_seq_add(st.mem, MAIN, h, h + n, d)
    lc.llama_memory_seq_add(st.mem, MAIN, h + d, h + d + n, -d)
    got = st._decode(tail, pos, MAIN, want_last=True)
    assert diff(got, ref) < 1e-3
    # and a one-way shift must actually change something, or the test above proves nothing
    _, pos = st.assemble(["NOTES"])
    lc.llama_memory_seq_add(st.mem, MAIN, h, h + n, d)
    moved = st._decode(tail, pos + d, MAIN, want_last=True)
    assert diff(moved, ref) > 1e-4


def test_order_changes_positions(st):
    """X+Y and Y+X must both run and place every token (no overlaps, no gaps)."""
    a, pa = st.assemble(["CREATE", "ZIP"])
    b, pb = st.assemble(["ZIP", "CREATE"])
    assert pa == pb == len(a) == len(b)
    out1, _ = st.run(["CREATE", "ZIP"], TAIL, max_tokens=8)
    out2, _ = st.run(["ZIP", "CREATE"], TAIL, max_tokens=8)
    assert isinstance(out1, str) and isinstance(out2, str)


def test_disk_roundtrip(st, tmp_path):
    a, _ = st.run(["CREATE", "ZIP"], TAIL, max_tokens=16)
    st.save_dir(tmp_path)
    st.header, st.modules = None, {}
    st.load_dir(tmp_path)
    b, _ = st.run(["CREATE", "ZIP"], TAIL, max_tokens=16)
    assert a == b


def test_multi_module_divergence_is_reported(st):
    """Not exact by design: later modules never saw earlier ones. Just measure it."""
    toks, pos = st.assemble(["CREATE", "NOTES", "ZIP"])
    tail = st.tok(TAIL)
    stitched = st._decode(tail, pos, MAIN, want_last=True)
    full = full_logits(st, toks + tail)
    top = lambda v: max(range(len(v)), key=v.__getitem__)
    print(f"\nmax |dlogit| = {diff(stitched, full):.4f}, same top token: {top(stitched) == top(full)}")


def test_all_fresh_is_exact(st):
    """fresh >= all module tokens: only the header is cached, which is exact."""
    full, _ = st.run_full(["CREATE", "NOTES", "ZIP"], TAIL, max_tokens=12)
    out, tm = st.run(["CREATE", "NOTES", "ZIP"], TAIL, max_tokens=12, fresh=10**6)
    assert tm.extra["fresh_modules"] == 3
    assert out == full


def test_fresh_last_module_only(st):
    k = st.split_fresh(["CREATE", "NOTES", "ZIP"], len(st.modules["ZIP"].tokens))
    assert k == 2


def test_fresh_idx_everything_is_exact(st):
    """Every module decoded fresh in place == the full prompt."""
    names = ["CREATE", "NOTES", "ZIP"]
    full, _ = st.run_full(names, TAIL, max_tokens=12)
    out, _ = st.run(names, TAIL, max_tokens=12, fresh_idx={0, 1, 2})
    assert out == full


def test_fresh_idx_middle_piece(st):
    """A fresh piece in the middle with cached pieces after it still assembles every token."""
    names = ["CREATE", "NOTES", "ZIP"]
    toks, pos = st.assemble(names, fresh_idx={1})
    assert pos == len(toks) == len(st.header.tokens) + sum(len(st.modules[n].tokens) for n in names)
