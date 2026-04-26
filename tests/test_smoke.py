"""Day 1/2 smoke tests — pure-import + parse + agreement logic, no network."""

from __future__ import annotations

import math

from sec_comment_letter_alpha import data_loader, features, llm, parse, pipeline, stats, universe
from sec_comment_letter_alpha.data_loader import FilingRecord
from sec_comment_letter_alpha.features import LLMFeature


def test_imports_resolve():
    for m in (data_loader, features, llm, parse, pipeline, stats, universe):
        assert m is not None


def test_llm_call_one_signature():
    # call_one is the canonical entrypoint replacing the broken shared retry
    assert callable(llm.call_one)
    assert "attempts" in llm.call_one.__code__.co_varnames


def test_split_segments_numbered():
    rec = FilingRecord(
        cik="0000320193",
        form="UPLOAD",
        accession="0000000000-00-000000",
        date="2020-01-15",
        text=(
            "Dear Apple,\n"
            "1. Please clarify your revenue recognition for services.\n"
            "2. Provide segment reporting reconciliation.\n"
            "3. Explain the goodwill impairment test.\n"
        ),
    )
    segs = parse.split_into_segments(rec)
    assert len(segs) >= 3
    assert any("revenue" in s.text.lower() for s in segs)


def test_split_segments_unnumbered():
    rec = FilingRecord(
        cik="0000320193", form="UPLOAD", accession="x", date="2020-01-15",
        text="Single block of comment letter text.",
    )
    segs = parse.split_into_segments(rec)
    assert len(segs) == 1


def test_response_lag():
    up = FilingRecord("c", "UPLOAD", "a1", "2020-01-15", "x")
    co = FilingRecord("c", "CORRESP", "a2", "2020-02-10", "y")
    assert parse.response_lag_days(up, co) == 26
    assert parse.response_lag_days(up, None) is None


def test_pair_upload_corresp():
    recs = [
        FilingRecord("c1", "UPLOAD", "a1", "2020-01-15", "x"),
        FilingRecord("c1", "CORRESP", "a2", "2020-02-10", "y"),
        FilingRecord("c2", "UPLOAD", "a3", "2020-03-01", "z"),
    ]
    pairs = parse.pair_upload_corresp(recs)
    assert len(pairs) == 2
    assert pairs[0][0].cik == "c1" and pairs[0][1] is not None
    assert pairs[1][0].cik == "c2" and pairs[1][1] is None


def test_parse_llm_json():
    out = features.parse_llm_json(
        'prefix {"topics": ["revenue_recognition"], "severity": 0.6, "resolution_signal": "ongoing"} suffix'
    )
    assert out["topics"] == ["revenue_recognition"]
    assert out["severity"] == 0.6


def test_topic_enum_size():
    assert len(features.TOPIC_ENUM) >= 10


def _f(cik, acc, topics, severity, resolution, model="m1"):
    return LLMFeature(
        cik=cik, accession=acc, date="2020-01-01",
        topics=topics, severity=severity, resolution_signal=resolution,
        response_lag_days=None, raw_model=model,
    )


def test_topic_jaccard_perfect_and_disjoint():
    a = {("c", "1"): _f("c", "1", ["revenue_recognition", "leases"], 0.5, "ongoing")}
    b = {("c", "1"): _f("c", "1", ["revenue_recognition", "leases"], 0.5, "ongoing")}
    assert features.topic_jaccard(a, b) == 1.0

    c = {("c", "1"): _f("c", "1", ["other"], 0.0, "unknown")}
    assert features.topic_jaccard(a, c) == 0.0


def test_severity_pearson_perfect():
    a = {("c", str(i)): _f("c", str(i), ["other"], i / 10, "unknown") for i in range(5)}
    b = {("c", str(i)): _f("c", str(i), ["other"], i / 10, "unknown") for i in range(5)}
    assert features.severity_pearson(a, b) > 0.99


def test_resolution_kappa_perfect():
    keys = [("c", str(i)) for i in range(4)]
    classes = ["accepted", "partial", "ongoing", "unknown"]
    a = {k: _f("c", k[1], ["other"], 0.0, classes[i]) for i, k in enumerate(keys)}
    b = {k: _f("c", k[1], ["other"], 0.0, classes[i]) for i, k in enumerate(keys)}
    assert features.resolution_kappa(a, b) == 1.0


def test_resolution_kappa_chance():
    # all same class on both sides → exp_agree=1.0 → kappa undefined → NaN
    keys = [("c", str(i)) for i in range(3)]
    a = {k: _f("c", k[1], ["other"], 0.0, "accepted") for k in keys}
    b = {k: _f("c", k[1], ["other"], 0.0, "accepted") for k in keys}
    assert math.isnan(features.resolution_kappa(a, b))


def test_pairwise_agreement_shape():
    seg_keys = [("c", str(i)) for i in range(4)]
    feats_per_model = {
        "m1": {k: _f("c", k[1], ["revenue_recognition"], 0.5, "ongoing", "m1") for k in seg_keys},
        "m2": {k: _f("c", k[1], ["revenue_recognition", "leases"], 0.6, "ongoing", "m2") for k in seg_keys},
    }
    results = [
        features.EnsembleResult(
            cik="c", accession=k[1], date="2020-01-01",
            per_model={m: feats_per_model[m][k] for m in ("m1", "m2")},
            errors={},
        )
        for k in seg_keys
    ]
    out = features.pairwise_agreement(results, models=("m1", "m2"))
    assert ("m1", "m2") in out
    metrics = out[("m1", "m2")]
    assert metrics["n_shared"] == 4
    assert 0.0 <= metrics["topic_jaccard"] <= 1.0


def test_extract_pdf_text_empty():
    assert parse.extract_pdf_text("") == ""


def test_extract_pdf_text_garbage():
    assert parse.extract_pdf_text("not a pdf at all") == ""


def test_extract_pdf_text_roundtrip():
    """Build a tiny in-memory PDF, round-trip through latin-1, verify no crash."""
    import io as _io

    try:
        from pypdf import PdfWriter
    except ImportError:
        return

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buf = _io.BytesIO()
    writer.write(buf)
    raw = buf.getvalue().decode("latin-1")
    out = parse.extract_pdf_text(raw)
    assert isinstance(out, str)


def test_split_segments_pdf_invalid_returns_empty():
    rec = FilingRecord(
        cik="c", form="UPLOAD", accession="a", date="2020-01-15",
        text="not a real pdf", primary="filing.pdf",
    )
    segs = parse.split_into_segments(rec)
    assert segs == []
