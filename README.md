# HAMix Quora

HAMix supports the study *The Engagement Dilemma in the Human-AI Coexistent Content Ecosystem: Empirical Evidence from the HAMix Quora Dataset* (GP3057) by Wang Xiang, Mingxuan Wu, and Peiang Yu.

## Reproduce the GP3057 revision

[Download the versioned GP3057 replication package](https://github.com/wxiang1-creator/HAMix-Quora/raw/refs/tags/gp3057-replication-v1/GP3057_Replication_Package.zip). Extract it into its own directory and run these commands from that directory with Python 3.11 or 3.12.

```sh
python -m pip install -r requirements.txt
python src/revision_checks.py
python src/analysis_engagement_models.py
python src/analysis_matching.py
python src/analysis_dimension.py
python src/generate_topconf_figures.py
```

| Script | Revised manuscript outputs |
| --- | --- |
| revision_checks.py | Alternative thresholds, equal interaction weights, and Tables 5, 6, and 8 |
| analysis_engagement_models.py | Baseline, question fixed-effects, count, and hurdle models in Tables 3–4 |
| analysis_matching.py | Matched comparisons in Table 4 |
| analysis_dimension.py | Quality regression in Table 7 |
| generate_topconf_figures.py | Grouped figures and summaries, including Figure 7 |

The package includes a data dictionary, the exact GPT-5-mini scoring prompt, and archived model and matching outputs. Analyses use frozen annotations and require no platform login or scoring API calls.

## Data and analysis samples

The replication extract contains 11,779 Quora answers from 2023–2026, with partial coverage in 2026. It includes pseudonymous question and author group codes, posting year, text length, recorded interactions and views, Winston AI scores and detector-derived labels, and five GPT-5-mini quality ratings with their unweighted mean. The extract omits answer text, question titles, URLs, and author names.

- Tables 3–4 use 11,735 common complete cases for the baseline and question fixed-effects models. The positive stage of the hurdle model uses 7,010 rows.
- Figure 7 and Table 5 use the same 11,286 answers with binary labels. The annual composition panel also includes 482 intermediate and 11 missing detector scores under the uncertain category.
- The primary thresholds are at least 0.8 for AI-like and at most 0.2 for human-like. The labels represent detector output. Authorship was unverified.
- Recorded engagement equals upvotes plus twice comments plus shares. Unavailable counters could be recorded as zero. The archive cannot separate these cases from observed zeros. The revision includes an equal-weight comparison.

The root data and source directories contain earlier development materials. Use the versioned package for the GP3057 results.
