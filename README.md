# HAMix: AI-like Content, Quality, and Engagement on Quora

## Overview
This repository contains the code, processed data, and analysis outputs for **HAMix**, a Quora-based dataset and research project on the relationship between **AI-like content**, **answer quality**, and **user engagement**.

The project investigates whether AI-like answers and human-like answers receive different levels of engagement on Quora, and whether this gap can be explained by observable factors such as text length and multi-dimensional answer quality.

The study uses:
- **Winston AI** for AI-likeness labeling
- **GPT-5-mini** for five-dimensional quality scoring
- descriptive statistics, visualization, and regression analysis for empirical evaluation

---

## Research Questions

### Primary Question
Do AI-like answers receive the same level of engagement as human-like answers on Quora?

### Follow-up Question
If not, can the difference be explained by answer quality, text length, or other observable content features?

---

## HAMix Dataset

HAMix was constructed from scraped Quora answers and processed through multiple stages.

### Dataset Pipeline
- **Initial scraped answers:** ~47,834
- **Final usable analytical samples:** ~11.8K

### Main Dataset Used in the Paper
The **final analytical version of HAMix used in the paper** is:

- `data/processed/final_analysis_dataset.csv`
  Cleaned archival version of the HAMix dataset preserved as the final processed release
  
### Other Related Data Files
- `data/processed/quora_answers_selenium.csv`  
  Intermediate scraped answer data

- `data/processed/quora_answers_cleaned.csv`  
  Cleaned answer data after preprocessing

- `data/processed/quora_answers_with_ai.csv`  
  Cleaned data with AI probability labels

- `data/processed/quora_analysis_dataset.csv`  
  Analysis-ready intermediate dataset

- `data/processed/merged.csv`  
  Main analysis dataset used for the descriptive and regression analyses in the paper

- `data/processed/robustness_results.csv`  
  Robustness analysis results under different AI-probability thresholds

---

## Variables in HAMix
The dataset includes variables such as:
- answer text
- posting year
- text length
- engagement
- AI probability
- AI-like / human-like labels
- quality scores:
  - clarity
  - completeness
  - relevance
  - originality
  - authenticity

### Engagement Definition
In this project, engagement is constructed as:

`engagement = upvotes + 2 × comments + shares`

Comments are weighted more heavily to reflect a stronger form of user interaction.

---

## Notes on Code and Comments

Some scripts in this repository contain Chinese comments or annotations from the original development process. These comments were kept for convenience during collaborative work and do not affect data processing, model scoring, or analysis results.

The repository is intended to preserve the actual project workflow rather than fully rewrite all internal comments into English.

---

## Reproducibility Notes
The Quora platform is dynamic, so newly scraped data may differ from the version used in the paper.
The analyses reported in the paper are based on the processed datasets stored in data/processed/.
The final paper mainly relies on final_analysis_dataset.csv as the final analytical version of HAMix.
Some intermediate CSV files were preserved to document the preprocessing and annotation pipeline.

---

## External Tools and Models

This project uses the following external tools:

Winston AI for AI-likeness probability labeling
GPT-5-mini for model-based quality scoring
statsmodels for regression analysis

These tools support the annotation and analysis pipeline, but all final interpretations and paper writing remain the responsibility of the authors.

---

## Security Note

Any API keys or private credentials used during development have been removed from the repository.
If environment-based access is needed, credentials should be supplied through local environment variables rather than hard-coded in source files.

---

## Authors
Wang Xiang
Mingxuan Wu
Peiang Yu

Affiliations in the paper:

College of Information, University of Maryland, College Park
Department of Information Management, Wuhan University
Department of Computer Science, American University

---

## Repository Structure

```bash
quora/
├── data/
│   ├── raw/                      # raw URLs and raw scraped files
│   ├── processed/                # cleaned, labeled, and analysis-ready datasets
│   └── outputs/
│       ├── descriptive_tables/   # descriptive statistics tables
│       ├── figures/              # plots and visualizations
│       └── regression_tables/    # regression output tables
│
├── src/
│   ├── scrape_quora_selenium.py      # scrape Quora answers from target pages
│   ├── clean_quora_data.py           # clean raw data and build processed datasets
│   ├── add_ai_label.py               # assign AI probability labels using Winston AI
│   ├── chat_Gpt_judge_five_dim.py    # score answers on five quality dimensions
│   ├── chat_gpt_judge.py             # earlier / auxiliary scoring script
│   ├── analysis_main.py              # main descriptive and regression analyses
│   ├── analysis_dimension.py         # quality-dimension analysis
│   ├── analysis_trend.py             # time-trend and temporal pattern analysis
│   └── login_once.py                 # one-time login helper for scraping
│
├── README.md
└── .gitignore



