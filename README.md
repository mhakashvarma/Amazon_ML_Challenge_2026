# Amazon ML Challenge 2026 — Business Entity Resolution

## Team: Tech Trojans

**Team Members**
- M H Akash Varma
- Mithrajith
- Lakshmeesha

---

## Project Overview

This project was developed for the **Amazon ML Challenge 2026 – Business Entity Resolution** problem.

The objective is to identify whether business entities appearing in different data sources refer to the same real-world business.

The solution uses a combination of:

- Text normalization
- Candidate generation / blocking
- String similarity
- Token-based similarity
- Address and name matching
- Machine learning-based candidate scoring
- CatBoost classification

The final solution was designed to handle a large number of business records while reducing unnecessary candidate comparisons.

---

## Solution Approach

The pipeline consists of the following major stages:

### 1. Data Preprocessing

Business names and addresses are normalized before matching.

The preprocessing includes:

- Lowercase conversion
- Unicode normalization
- Punctuation normalization
- Whitespace normalization
- Common address abbreviation normalization

Examples of address normalization include:

- `street → st`
- `road → rd`
- `avenue → ave`
- `drive → dr`
- `boulevard → blvd`
- `highway → hwy`
- `lane → ln`
- `parkway → pkwy`
- `place → pl`
- `suite → ste`

---

### 2. Candidate Generation

Instead of comparing every source business with every target business, blocking is used to generate a smaller set of plausible candidates.

The blocking keys include:

- Name 5-character prefix
- Name 4-character prefix
- Last 5 characters of name
- First name token
- Address 15-character prefix
- Address 10-character prefix
- Address 5-character prefix
- First address token
- Last address token
- House number
- Address component

A bucket-size limit was also used to prevent extremely large candidate groups.

---

### 3. Feature Engineering

Candidate pairs are represented using name, address, and structural similarity features.

Important features include:

- Exact name match
- Exact address match
- Exact country match
- Name similarity
- Address similarity
- Name token overlap
- Address token overlap
- Name Jaccard similarity
- Address Jaccard similarity
- Shared name token count
- Shared address token count
- Shared address number count
- Name length difference
- Address length difference
- Name length ratio
- Address length ratio
- Similarity gap
- Maximum similarity
- Similarity product
- Token overlap gap

---

## Matching Model

A **CatBoost classifier** was used to score candidate entity pairs.

The final V5 model uses **20 engineered features** derived from business names, addresses, countries, token overlap, and string similarity.

A matching threshold of **0.60** was used for the V5 rescue matching stage.

---

## Submission Strategy

The final submission strategy started from the baseline matching results and focused additional matching effort on Source-1 entities that were not matched by the baseline.

The rescue stage generated additional candidate pairs and used the V5 CatBoost model to identify additional likely matches.

The final candidate set was constructed as the union of the baseline candidate set and rescue candidate set.

---

## Results

The final Submission B contained:

| Metric | Value |
|---|---:|
| Source-1 rows | 1,732,544 |
| Baseline non-empty matches | 1,051,190 |
| New rescue matches | 329,571 |
| Final non-empty matches | 1,380,761 |
| Final empty matches | 351,783 |
| Candidate rows | 1,732,544 |
| Candidate file non-empty rows | 1,727,168 |
| Candidate violations | 0 |

The official candidate/matching audit found **0 candidate-set violations**.

The final submission achieved a **leaderboard score of 0.558**.

The earlier baseline submission achieved **0.463**.

---

## Repository Structure

```text
Amazon_ML_Challenge_2026/
│
├── Amazon_ML_Challenge_2026.ipynb
├── README.md
│
└── V5_ML_Final/
    ├── V5_final_report.txt
    ├── v5_feature_list.txt
    ├── v5_inference.py
    └── v5_model.cbm
