from summft.metrics import rouge_scores


def test_identical_is_one():
    s = rouge_scores(["Amanda baked cookies"], ["Amanda baked cookies"])
    assert all(abs(v - 1.0) < 1e-9 for v in s.values())

def test_no_overlap_is_zero():
    s = rouge_scores(["Amanda baked cookies"], ["dog ran fast"])
    assert all(v == 0.0 for v in s.values())