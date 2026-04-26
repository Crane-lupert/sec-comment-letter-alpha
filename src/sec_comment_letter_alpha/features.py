"""LLM feature extraction for UPLOAD/CORRESP segments.

Three-model ensemble (Gemma + Llama + Claude) per CLAUDE.md Day 2-3 plan.
LLM transport lives in `.llm` (workaround for the shared retry bug; see that
module's docstring).

Output schema (one row per UPLOAD letter):
{
    "filing_date": "YYYY-MM-DD",
    "cik": "<cik10>",
    "topics": [str, ...],          # multi-label, ~15 enum
    "severity": float,             # 0..1
    "response_lag_days": int|null,
    "resolution_signal": "accepted|partial|ongoing|unknown",
}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from statistics import mean

from shared_utils.openrouter_client import OpenRouterClient

from .llm import call_one
from .parse import ParsedSegment


PROJECT_TAG = "X"

TOPIC_ENUM = [
    "revenue_recognition",
    "segment_reporting",
    "non_gaap_metrics",
    "goodwill_impairment",
    "income_taxes",
    "fair_value",
    "internal_controls",
    "business_combinations",
    "stock_compensation",
    "leases",
    "loss_contingencies",
    "going_concern",
    "related_party",
    "cyber_disclosure",
    "other",
]

PROMPT_V1 = """You are an SEC disclosure analyst. Read the SEC comment letter excerpt and emit STRICT JSON.

Schema:
{{
  "topics": [<one or more from {topics}>],
  "severity": <float 0..1; 0=editorial, 1=going-concern/restatement-grade>,
  "resolution_signal": "<accepted|partial|ongoing|unknown>"
}}

Output JSON only, no prose, no fences.

Excerpt:
\"\"\"{excerpt}\"\"\"
"""

PROMPT_V2 = """You are an SEC disclosure analyst. Read the excerpt below and emit STRICT JSON describing what the LETTER ITSELF SAYS — do not infer beyond the excerpt.

Schema:
{{
  "topics": [<one or more from {topics}>],
  "severity": <float 0..1>,
  "resolution_signal": "<accepted|partial|ongoing|unknown>"
}}

severity rubric (anchor to the nearest band):
  0.0-0.2: editorial / formatting / wording clarification, no accounting impact
  0.2-0.5: disclosure-improvement request, no restatement implication
  0.5-0.8: substantive accounting question potentially requiring a disclosure
           change (revenue recognition, segment, fair value, goodwill)
  0.8-1.0: restatement, going-concern, material weakness, fraud-adjacent

resolution_signal — choose ONE based ONLY on THIS letter:
  "accepted":  letter EXPLICITLY closes the review or states no further comments.
               Anchors: "we have completed our review", "no further comments",
                         "this concludes our review".
  "partial":   letter acknowledges the registrant's prior response and either
               accepts some points while pursuing others, OR responds to
               specific prior comments.
               Anchors: "we have considered your response to comment N and ...",
                         "based on your response, please ...".
  "ongoing":   letter raises new questions or follow-ups without indicating
               closure. DEFAULT for first-round letters and any letter whose
               primary act is asking new questions.
               Anchors: "please provide", "tell us", "we note ... please explain".
  "unknown":   excerpt too short, malformed, non-substantive (cover sheet only,
               parsing failure), or genuinely ambiguous between the above.

Output STRICT JSON only. No prose, no fences, no comments inside the JSON.

Excerpt:
\"\"\"{excerpt}\"\"\"
"""

PROMPT_V3_CORRESP = """You are an SEC disclosure analyst reading a CORRESP letter (a registrant's RESPONSE to a prior SEC comment letter). Emit STRICT JSON describing what THIS LETTER SAYS — do not infer beyond the excerpt.

Schema:
{{
  "topics": [<one or more from {topics}>],
  "severity": <float 0..1>,
  "response_intent": "<agree_revise|explain_position|supplemental|pushback|closing>"
}}

severity rubric (anchor to nearest band; reflects underlying issue depth, not response tone):
  0.0-0.2: editorial / formatting / wording clarification
  0.2-0.5: disclosure-improvement matter, no restatement implication
  0.5-0.8: substantive accounting question (revenue recognition, segment, fair value, goodwill)
  0.8-1.0: restatement, going-concern, material weakness, fraud-adjacent

response_intent (registrant's primary speech act in THIS letter; pick the SINGLE dominant one):
  "agree_revise":     registrant commits to modify a future filing or has filed a revised exhibit.
                      Anchors: "we will revise in our next amendment", "please find our revised draft",
                                "we have updated our disclosure to ...", "we will include ... in future filings".
  "explain_position": registrant defends current treatment as correct, citing standards or facts.
                      Anchors: "our current accounting is consistent with ASC ...", "we believe our prior
                      disclosure complies with ...", "the registrant respectfully maintains that ...".
  "supplemental":     registrant provides additional information requested without changing position or filing.
                      Anchors: "in response to comment N, we supplementally advise that ...",
                      "for your information, ...", "the company is providing the following information".
  "pushback":         registrant explicitly disagrees with the SEC's interpretation or contests a comment.
                      Anchors: "we respectfully disagree", "we do not believe ... is required",
                      "the staff's position is inconsistent with ...".
  "closing":          registrant confirms all comments are addressed, no further action expected.
                      Anchors: "we believe this addresses all of the staff's comments",
                      "we trust this is responsive", "no further amendments are anticipated".

If the letter mixes multiple intents across different comments, pick the dominant act for the LETTER as a whole, weighted by paragraph count. If genuinely impossible to pick, you may emit "supplemental" as the conservative default (do NOT use "unknown").

Output STRICT JSON only. No prose, no fences, no comments inside the JSON.

Excerpt:
\"\"\"{excerpt}\"\"\"
"""

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2, "v3-corresp": PROMPT_V3_CORRESP}
DEFAULT_PROMPT_VERSION = "v2"

# Backwards-compat: any caller that imported PROMPT_TEMPLATE still gets the
# canonical v1 string (the one used to produce data/day2_ensemble.json).
PROMPT_TEMPLATE = PROMPT_V1


@dataclass(frozen=True)
class LLMFeature:
    cik: str
    accession: str
    date: str
    topics: list[str]
    severity: float
    resolution_signal: str
    response_lag_days: int | None
    raw_model: str


def build_prompt(
    segment: ParsedSegment,
    max_chars: int = 6000,
    *,
    version: str = DEFAULT_PROMPT_VERSION,
) -> str:
    if version not in PROMPTS:
        raise ValueError(f"unknown prompt version {version!r}; choose from {list(PROMPTS)}")
    excerpt = (segment.text or "").strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars] + " ...[truncated]"
    return PROMPTS[version].format(topics=TOPIC_ENUM, excerpt=excerpt)


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_llm_json(content: str) -> dict:
    """Extract first JSON object from LLM output, tolerating wrapping prose."""
    m = _JSON_BLOCK.search(content or "")
    if not m:
        raise ValueError(f"no JSON object in LLM output: {content!r}")
    return json.loads(m.group(0))


ENSEMBLE_DEFAULT: tuple[str, ...] = (
    "google/gemma-3-27b-it",
    "meta-llama/llama-3.3-70b-instruct",
    "anthropic/claude-haiku-4.5",
)


def extract_one(
    client: OpenRouterClient,
    segment: ParsedSegment,
    *,
    model: str = "google/gemma-3-27b-it",
    response_lag_days: int | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> LLMFeature:
    prompt = build_prompt(segment, version=prompt_version)
    resp = call_one(client, model=model, prompt=prompt, max_tokens=400, temperature=0.0)
    content = resp["choices"][0]["message"]["content"]
    parsed = parse_llm_json(content)

    topics = parsed.get("topics") or ["other"]
    if isinstance(topics, str):
        topics = [topics]
    topics = [t for t in topics if t in TOPIC_ENUM] or ["other"]

    # v1/v2 emit "resolution_signal"; v3-corresp emits "response_intent".
    # We store both in resolution_signal field (the categorical column);
    # the prompt version recorded in the extraction output disambiguates semantics.
    cat = parsed.get("resolution_signal") or parsed.get("response_intent") or "unknown"
    return LLMFeature(
        cik=segment.cik,
        accession=segment.accession,
        date=segment.date,
        topics=topics,
        severity=float(parsed.get("severity", 0.0)),
        resolution_signal=str(cat),
        response_lag_days=response_lag_days,
        raw_model=model,
    )


@dataclass(frozen=True)
class EnsembleResult:
    """One segment scored by N models. `errors[model] = exception_str` for failures."""
    cik: str
    accession: str
    date: str
    per_model: dict[str, LLMFeature]
    errors: dict[str, str] = field(default_factory=dict)


def extract_ensemble(
    client: OpenRouterClient,
    segment: ParsedSegment,
    *,
    models: tuple[str, ...] = ENSEMBLE_DEFAULT,
    response_lag_days: int | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> EnsembleResult:
    """Score one segment with each of `models`. Per-model failures captured, not raised."""
    per_model: dict[str, LLMFeature] = {}
    errors: dict[str, str] = {}
    for m in models:
        try:
            per_model[m] = extract_one(
                client, segment, model=m, response_lag_days=response_lag_days,
                prompt_version=prompt_version,
            )
        except Exception as e:  # noqa: BLE001
            errors[m] = f"{type(e).__name__}: {e}"
    return EnsembleResult(
        cik=segment.cik,
        accession=segment.accession,
        date=segment.date,
        per_model=per_model,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Inter-model agreement metrics
# ---------------------------------------------------------------------------


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def topic_jaccard(features_a: dict[str, LLMFeature], features_b: dict[str, LLMFeature]) -> float:
    """Mean Jaccard over the (cik, accession) keys in both dicts."""
    keys = set(features_a) & set(features_b)
    if not keys:
        return float("nan")
    return mean(_jaccard(features_a[k].topics, features_b[k].topics) for k in keys)


def severity_pearson(features_a: dict[str, LLMFeature], features_b: dict[str, LLMFeature]) -> float:
    """Pearson correlation of severity over shared keys. NaN when <3 pairs or zero variance."""
    keys = sorted(set(features_a) & set(features_b))
    if len(keys) < 3:
        return float("nan")
    xs = [features_a[k].severity for k in keys]
    ys = [features_b[k].severity for k in keys]
    n = len(keys)
    mx, my = mean(xs), mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / (sxx * syy) ** 0.5


def resolution_kappa(features_a: dict[str, LLMFeature], features_b: dict[str, LLMFeature]) -> float:
    """Cohen's κ on resolution_signal (4-class: accepted/partial/ongoing/unknown)."""
    keys = sorted(set(features_a) & set(features_b))
    if not keys:
        return float("nan")
    classes = ["accepted", "partial", "ongoing", "unknown"]
    n = len(keys)
    obs_agree = sum(
        1 for k in keys
        if features_a[k].resolution_signal == features_b[k].resolution_signal
    ) / n
    pa = {c: sum(1 for k in keys if features_a[k].resolution_signal == c) / n for c in classes}
    pb = {c: sum(1 for k in keys if features_b[k].resolution_signal == c) / n for c in classes}
    exp_agree = sum(pa[c] * pb[c] for c in classes)
    if exp_agree >= 1.0:
        return float("nan")
    return (obs_agree - exp_agree) / (1.0 - exp_agree)


def pairwise_agreement(
    results: list[EnsembleResult],
    *,
    models: tuple[str, ...] = ENSEMBLE_DEFAULT,
) -> dict[tuple[str, str], dict[str, float]]:
    """Compute Jaccard / Pearson / κ for every model pair across `results`.

    Skips ensemble rows where either model errored. Returns dict keyed by
    sorted (model_a, model_b) tuples.
    """
    by_model: dict[str, dict[str, LLMFeature]] = {m: {} for m in models}
    for r in results:
        for m, feat in r.per_model.items():
            by_model.setdefault(m, {})[(r.cik, r.accession)] = feat  # type: ignore[index]
    out: dict[tuple[str, str], dict[str, float]] = {}
    for i, ma in enumerate(models):
        for mb in models[i + 1:]:
            out[(ma, mb)] = {
                "topic_jaccard": topic_jaccard(by_model[ma], by_model[mb]),
                "severity_pearson": severity_pearson(by_model[ma], by_model[mb]),
                "resolution_kappa": resolution_kappa(by_model[ma], by_model[mb]),
                "n_shared": len(set(by_model[ma]) & set(by_model[mb])),
            }
    return out
