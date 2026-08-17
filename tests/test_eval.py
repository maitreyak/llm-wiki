from llm_wiki.eval.data import EvalData, EvalItem
from llm_wiki.eval.metrics import best_scores, exact_match, f1_score, normalize_answer
from llm_wiki.eval.run import QuestionResult, corpus_hash, summarize


def test_normalize():
    assert normalize_answer("The  Quick, Brown Fox!") == "quick brown fox"
    assert normalize_answer("A dog") == "dog"


def test_f1_and_em():
    assert f1_score("Barack Obama", "Barack Obama") == 1.0
    assert exact_match("the Barack Obama", "Barack Obama.") == 1.0
    assert 0 < f1_score("Barack Hussein Obama", "Barack Obama") < 1.0
    assert f1_score("Paris", "London") == 0.0
    # yes/no strictness
    assert f1_score("yes it is", "yes") == 0.0
    assert f1_score("yes", "yes") == 1.0


def test_best_scores_over_aliases():
    f1, em = best_scores("NYC", ["New York City", "NYC"])
    assert f1 == 1.0 and em == 1.0
    assert best_scores("x", []) == (0.0, 0.0)


def test_corpus_dedup_and_hash():
    items = [
        EvalItem("1", "q1", ["a"], [("T1", "text1"), ("T2", "text2")]),
        EvalItem("2", "q2", ["a"], [("T2", "text2"), ("T3", "text3")]),
    ]
    data = EvalData("d", items)
    corpus = data.corpus(2)
    assert [t for t, _ in corpus] == ["T1", "T2", "T3"]
    assert data.corpus(1) == [("T1", "text1"), ("T2", "text2")]
    h1 = corpus_hash(corpus)
    assert h1 == corpus_hash(list(corpus))
    assert h1 != corpus_hash(corpus[:2])


def qr(f1, em, qtype="", hops=None, calls=3):
    return QuestionResult(
        qid="q", question="?", gold="g", prediction="p",
        f1=f1, em=em, qtype=qtype, hops=hops,
        tool_calls=calls, stop_reason="answered", seconds=1.0,
    )


def test_summarize_breakdowns():
    results = [qr(1.0, 1.0, qtype="bridge", hops=2), qr(0.5, 0.0, qtype="comparison", hops=4)]
    s = summarize(results)
    assert s["f1"] == 0.75 and s["em"] == 0.5 and s["n"] == 2
    assert s["by_type"]["bridge"]["f1"] == 1.0
    assert s["by_hops"]["4"]["f1"] == 0.5
    assert summarize([]) == {"f1": 0.0, "em": 0.0, "n": 0}
