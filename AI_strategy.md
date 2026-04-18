# Silver Economy AI Copilot — Implemented Strategy

This app includes an AI layer implemented locally inside Streamlit, without requiring a separate LLM service.

## What is implemented

### 1) Market brief generator
The app can generate a plain-English county brief from the active dashboard state. It explains:
- why a county is strong or weak
- how Silver Score compares with NASI
- which components are helping or hurting
- what the next validation step should be

### 2) Scenario assistant
Users can type prompts like:
- "Top 10 affordable, lower-risk senior markets in Virginia"
- "Top 8 senior markets with strong income"
- "Top 12 Phase 1 markets in Florida"

The app parses those prompts and converts them into ranking emphasis, state filters, and output counts.

### 3) County comparison assistant
Users can select two counties and get an AI-style explanation of which one is stronger overall and why.

### 4) Data-quality explainer
The app summarizes which fields are live, which are fallback proxies, and whether key columns have missing values.

### 5) Recommendation memory
Users can save the active screening setup during the current session and export saved scenarios as CSV.

---

## Why this approach for AI for the MVP
This version uses deterministic, explainable decision logic instead of an external generative model. That means:
- it works offline
- it is reproducible
- it can be defended in class
- it creates a clean contract for a future LLM-backed version

---

## Next upgrade path
A later version can replace the local text templates and parser with an API-backed LLM service.

### Future endpoint shape
`POST /ai/market-brief`

Payload:
- current filters
- current weights
- selected counties
- top ranked table
- score decomposition
- data quality notes

### Future LLM responsibilities
- richer executive summaries
- more flexible natural-language querying
- investment memo drafting
- presentation-ready county comparison writeups
- follow-up Q&A against the active dashboard state
