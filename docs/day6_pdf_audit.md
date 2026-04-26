# Day 6 — PDF Extraction Quality Audit

## Why

Roughly half of cached SEC UPLOAD/CORRESP filings arrive as PDFs. Post the
daemon-side latin-1 byte-preservation patch, `parse.extract_pdf_text`
returns non-empty strings on ~99% of PDFs, but "non-empty" does not imply
"semantically correct" — pypdf can mangle tables, footnotes, multi-column
layouts, and OCR-scanned letters. This audit measures actual extraction
quality on a random sample so we know whether to trust PDF text downstream
of the LLM feature pipeline.

## Method

1. Enumerated all cached UPLOAD filings whose `primary` ends in `.pdf`,
   restricted to the R3K universe (`data/universe_ciks_r3k.parquet`,
   19568 eligible PDF UPLOADs).
2. Drew a random sample of n=30 (seed=42).
3. For each, called `parse.extract_pdf_text(rec.text)` (the same code path
   used by `parse.split_into_segments`, no modifications), then computed
   six text-quality metrics:
   - **n_words** — substantive content proxy.
   - **sentence_rate** — sentence-ending punctuation per 100 words; English
     prose is typically 3-8 (Flesch / Hunt 1965 style).
   - **ws_ratio** — whitespace chars / total chars; clean prose is ~0.10-0.22.
   - **alpha_ratio** — alphabetic chars / total chars; clean English is
     ~0.65-0.85.
   - **printable_ratio** — printable chars / total; mojibake drags this
     below ~0.95.
   - **mean_token_len** — average token length in chars; garbled PDFs often
     show tons of 1-char tokens or massive run-on tokens.
4. Combined the six metrics into a `confidence ∈ [0,1]` weighted blend
   (see `_score` in `scripts/day6_pdf_audit.py`) and a categorical
   label `full | partial | garbled | empty`:
   - **empty**: 0 words extracted.
   - **garbled**: <50 words OR confidence <0.45.
   - **partial**: confidence in [0.55, 0.80).
   - **full**: confidence ≥0.80 and n_words ≥200.
5. No SEC re-fetch (project rule: never call SEC directly). The audit is
   self-consistent on cached extractions only.

## Results

### Distribution

| score | count | pct |
|-------|-------|-----|
| full     | 13 | 43.3% |
| partial  | 17 | 56.7% |
| garbled  | 0 | 0.0% |
| empty    | 0 | 0.0% |

- Mean confidence (all rows): **0.947**
- Mean confidence by label: full=1.000 | partial=0.906 | garbled=0.000 | empty=0.000
- n_words quartiles: p25=99 | p50=170 | p75=595 | min=89 | max=1090

### Confidence histogram

| bin | range | count | bar |
|----|------|------|-----|
| 0 | [0.0, 0.1) | 0 | `` |
| 1 | [0.1, 0.2) | 0 | `` |
| 2 | [0.2, 0.3) | 0 | `` |
| 3 | [0.3, 0.4) | 0 | `` |
| 4 | [0.4, 0.5) | 0 | `` |
| 5 | [0.5, 0.6) | 0 | `` |
| 6 | [0.6, 0.7) | 0 | `` |
| 7 | [0.7, 0.8) | 0 | `` |
| 8 | [0.8, 0.9) | 9 | `#########` |
| 9 | [0.9, 1.0) | 21 | `####################` |

### All sampled rows (sorted by confidence asc)

| cik | accession | primary | n_words | sent_rate | ws_ratio | alpha | conf | score |
|------|-----------|---------|---------|-----------|----------|-------|------|-------|
| 0001018979 | 0000000000-07-048529 | filename1.pdf | 94 | 9.57 | 0.194 | 0.670 | 0.814 | partial |
| 0001971213 | 0000000000-24-007328 | filename1.pdf | 106 | 2.83 | 0.156 | 0.758 | 0.826 | partial |
| 0000051434 | 0000000000-24-009831 | filename1.pdf | 89 | 5.62 | 0.149 | 0.759 | 0.889 | partial |
| 0000930236 | 0000000000-25-009915 | filename1.pdf | 91 | 7.69 | 0.158 | 0.751 | 0.891 | partial |
| 0001609711 | 0000000000-22-013700 | filename1.pdf | 93 | 6.45 | 0.154 | 0.754 | 0.893 | partial |
| 0000049754 | 0000000000-17-038298 | filename1.pdf | 95 | 7.37 | 0.154 | 0.745 | 0.895 | partial |
| 0001714899 | 0000000000-19-014345 | filename1.pdf | 96 | 7.29 | 0.155 | 0.754 | 0.896 | partial |
| 0000008818 | 0000000000-23-005412 | filename1.pdf | 97 | 5.16 | 0.151 | 0.763 | 0.897 | partial |
| 0001567683 | 0000000000-21-011430 | filename1.pdf | 99 | 6.06 | 0.154 | 0.754 | 0.899 | partial |
| 0000040987 | 0000000000-23-009088 | filename1.pdf | 100 | 3.00 | 0.153 | 0.765 | 0.900 | partial |
| 0000726958 | 0000000000-23-003652 | filename1.pdf | 100 | 6.00 | 0.155 | 0.756 | 0.900 | partial |
| 0001817358 | 0000000000-22-013982 | filename1.pdf | 101 | 4.95 | 0.155 | 0.757 | 0.901 | partial |
| 0001818093 | 0000000000-22-009309 | filename1.pdf | 129 | 6.98 | 0.157 | 0.757 | 0.929 | partial |
| 0001157408 | 0000000000-14-039325 | filename1.pdf | 159 | 5.03 | 0.169 | 0.759 | 0.959 | partial |
| 0000752714 | 0000000000-14-029352 | filename1.pdf | 170 | 6.47 | 0.164 | 0.769 | 0.970 | partial |
| 0001015820 | 0000000000-15-052987 | filename1.pdf | 170 | 5.88 | 0.167 | 0.773 | 0.970 | partial |
| 0000045876 | 0000000000-16-061474 | filename1.pdf | 170 | 3.53 | 0.163 | 0.778 | 0.970 | partial |
| 0000788965 | 0000000000-09-068459 | filename1.pdf | 863 | 4.75 | 0.166 | 0.775 | 1.000 | full |
| 0001138118 | 0000000000-17-016618 | filename1.pdf | 610 | 4.59 | 0.167 | 0.775 | 1.000 | full |
| 0001062231 | 0000000000-17-001358 | filename1.pdf | 667 | 6.30 | 0.157 | 0.783 | 1.000 | full |
| 0001022671 | 0000000000-21-013197 | filename1.pdf | 467 | 6.21 | 0.151 | 0.793 | 1.000 | full |
| 0000854775 | 0000000000-15-013992 | filename1.pdf | 892 | 5.94 | 0.156 | 0.799 | 1.000 | full |
| 0000713676 | 0000000000-23-014101 | filename1.pdf | 944 | 4.45 | 0.152 | 0.800 | 1.000 | full |
| 0001042893 | 0000000000-24-008302 | filename1.pdf | 383 | 5.22 | 0.158 | 0.759 | 1.000 | full |
| 0001835856 | 0000000000-21-000637 | filename1.pdf | 493 | 5.27 | 0.155 | 0.786 | 1.000 | full |
| 0001562476 | 0000000000-15-018006 | filename1.pdf | 808 | 5.82 | 0.158 | 0.792 | 1.000 | full |
| 0001964738 | 0000000000-23-007285 | filename1.pdf | 1090 | 4.77 | 0.154 | 0.796 | 1.000 | full |
| 0000885978 | 0000000000-17-001532 | filename1.pdf | 550 | 7.46 | 0.154 | 0.777 | 1.000 | full |
| 0001568100 | 0000000000-19-004084 | filename1.pdf | 652 | 5.52 | 0.157 | 0.784 | 1.000 | full |
| 0001377789 | 0000000000-19-010302 | filename1.pdf | 336 | 6.84 | 0.157 | 0.750 | 1.000 | full |

## Highlighted samples (verbatim text — human eyeball check)

### CIK 0001022671 — accession 0000000000-21-013197

- date: 2021-10-29
- primary: filename1.pdf
- n_words: 467 | sentence_rate: 6.21 | alpha_ratio: 0.793 | ws_ratio: 0.151
- confidence: 1.000 | score: **full**

First 1000 chars (verbatim, for human eyeball):

```
United States securities and exchange commission logo October 29, 2021 Theresa E. Wagler Chief Financial Officer Steel Dynamics, Inc. 7575 West Jefferson Blvd Fort Wayne, IN 46804 Re: Steel Dynamics, Inc. Form 10-K for the Fiscal Year Ended December 31, 2020 Response Dated October 14, 2021 File No. 000-21719 Dear Ms. Wagler: We have reviewed your October 14, 2021, response to our comment letter and have the following comments. In some of our comments, we may ask you to provide us with information so we may better understand your disclosure. Please respond to these comments within ten business days by providing the requested information or advise us as soon as possible when you will respond. If you do not believe our comments apply to your facts and circumstances, please tell us why in your response. After reviewing your response to these comments, we may have additional comments. Unless we note otherwise, our references to prior comments are to comments in our September 23, 2021, lette
```

### CIK 0000752714 — accession 0000000000-14-029352

- date: 2014-06-10
- primary: filename1.pdf
- n_words: 170 | sentence_rate: 6.47 | alpha_ratio: 0.769 | ws_ratio: 0.164
- confidence: 0.970 | score: **partial**

First 1000 chars (verbatim, for human eyeball):

```
UNITED STATES SECURITIES AND EXCHANGE COMMISSION WASHINGTON, D.C. 20549-4631 DIVISION OF CORPORATION FINANCE June 10, 2014 Via Facsimile Mr. Keith E. Pratt Chief Financial Officer McGrath Rentcorp 5700 Las Positas Road Livermore, CA 94551-7800 Re: McGrath Rentcorp Definitive Proxy on Form 14A Filed April 30, 2014 File No. 0-13292 Dear Mr. Pratt: We have completed our review of your filing. We remind you that our comments or changes to disclosure in response to our comments do not foreclose the Commission from taking any action with respect to the company or the filing and the company may not assert staff comments as a defense in any proceeding initiated by the Commission or any person under the federal securities laws of the United States. We urge all persons who are responsible for the accuracy and adequacy of the disclosure in the filing to be certain that the filing includes the information the Securities Exchange Act of 1934 and all applicable rules require. Sincerely, /s/ W. John 
```

### CIK 0000713676 — accession 0000000000-23-014101

- date: 2023-12-26
- primary: filename1.pdf
- n_words: 944 | sentence_rate: 4.45 | alpha_ratio: 0.800 | ws_ratio: 0.152
- confidence: 1.000 | score: **full**

First 1000 chars (verbatim, for human eyeball):

```
United States securities and exchange commission logo December 26, 2023 Robert Reilly Chief Financial Officer PNC Financial Services Group, Inc. The Tower at PNC Plaza, 300 Fifth Avenue Pittsburgh, Pennsylvania 15222-2401 Re: PNC Financial Services Group, Inc. Form 10-K for the Fiscal Year Ended December 31, 2022 File No. 001-09718 Dear Robert Reilly: We have reviewed your filing and have the following comments. Please respond to this letter within ten business days by providing the requested information or advise us as soon as possible when you will respond. If you do not believe a comment applies to your facts and circumstances, please tell us why in your response. After reviewing your response to this letter, we may have additional comments. Form 10-K for the Fiscal Year Ended December 31, 2022 Item 7 - Management's Discussion and Analysis of Financial Condition and results of Operations Funding Sources, page 47 1. We note your disclosure regarding the composition and change in your
```

### CIK 0001567683 — accession 0000000000-21-011430

- date: 2021-09-21
- primary: filename1.pdf
- n_words: 99 | sentence_rate: 6.06 | alpha_ratio: 0.754 | ws_ratio: 0.154
- confidence: 0.899 | score: **partial**

First 1000 chars (verbatim, for human eyeball):

```
United States securities and exchange commission logo September 21, 2021 Chad Plotkin Chief Financial Officer Clearway Energy, Inc. 300 Carnegie Center Suite 300 Princeton, NJ 08540 Re: Clearway Energy, Inc. Form 10-K for the Fiscal Year ended December 31, 2020 Filed March 1, 2021 File No. 001-36002 Dear Mr. Plotkin: We have completed our review of your filing. We remind you that the company and its management are responsible for the accuracy and adequacy of their disclosures, notwithstanding any review, comments, action or absence of action by the staff. Sincerely, Division of Corporation Finance Office of Energy & Transportation
```

### CIK 0001964738 — accession 0000000000-23-007285

- date: 2023-07-10
- primary: filename1.pdf
- n_words: 1090 | sentence_rate: 4.77 | alpha_ratio: 0.796 | ws_ratio: 0.154
- confidence: 1.000 | score: **full**

First 1000 chars (verbatim, for human eyeball):

```
United States securities and exchange commission logo July 7, 2023 Jeffrey Lavers President 3M Health Care Co 3M Center St. Paul , Minnesota 55144 Re: 3M Health Care Co Amendment No. 2 to Draft Registration Statement on Form 10-12G Submitted June 23, 2023 CIK No. 0001964738 Dear Jeffrey Lavers: We have reviewed your amended draft registration statement and have the following comments. In some of our comments, we may ask you to provide us with information so we may better understand your disclosure. Please respond to this letter by providing the requested information and either submitting an amended draft registration statement or publicly filing your registration statement on EDGAR. If you do not believe our comments apply to your facts and circumstances or do not believe an amendment is appropriate, please tell us why in your response. After reviewing the information you provide in response to these comments and your amended draft registration statement or filed registration statement
```


## Verdict

TRUSTABLE for downstream LLM (full+partial >= 80%, garbled <= 10%).

## Caveats

- Heuristic-only: no ground-truth comparison. A high-confidence row with
  pristine prose can still have wrong table values or mis-ordered
  multi-column content.
- The R3K filter is naive (no PIT membership), so a few sampled CIKs may
  not have been in R3K on the filing date. Acceptable for an extraction-
  quality audit.
- pypdf's `extract_text` is page-oriented; we lose page boundaries by
  joining with spaces. Downstream LLM is order-tolerant, so this is fine
  for topic/severity but would break for any positional analysis.

## Reproduce

```
.venv/Scripts/python.exe scripts/day6_pdf_audit.py --n 30 --seed 42
```
