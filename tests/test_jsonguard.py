"""JsonGuard: picks the best token that keeps the output compact, matched JSON."""
from kvstitch.jsonguard import JsonGuard

VOCAB = ["{", "}", "[", "]", '"plan"', ":", ",", ' "a b"', '"a b"', "\n", "  ", "Here", '{"x":1}', "]}"]


def run(prefs):
    """prefs: list of token lists, best first, one per step; returns the text written."""
    g = JsonGuard(VOCAB.__getitem__, top_k=len(VOCAB))
    out = ""
    for order in prefs:
        logits = [0.0] * len(VOCAB)
        for rank, tok in enumerate(order):
            logits[VOCAB.index(tok)] = 100.0 - rank
        t = g.pick(logits)
        if t is None:
            break
        out += VOCAB[t]
        if g.done:
            break
    return out, g


def test_skips_whitespace_and_text_before_the_object():
    out, _ = run([["Here", "\n", "{"], ["\n", '"plan"'], ["  ", ":"], ["["], ["]"], ["}"]])
    assert out == '{"plan":[]}'


def test_space_inside_a_string_is_kept_outside_is_not():
    out, _ = run([["{"], [' "a b"', '"a b"'], [":"], ['"a b"'], ["}"]])
    assert out == '{"a b":"a b"}'


def test_brackets_must_match_and_the_answer_ends_when_the_object_closes():
    out, g = run([["{"], ['"plan"'], [":"], ["["], ["}", "]"], ["}"], ["{"]])
    assert out == '{"plan":[]}' and g.done


def test_multi_char_token_checked_as_a_whole():
    out, g = run([['{"x":1}']])
    assert out == '{"x":1}' and g.done
