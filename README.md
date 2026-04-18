# Silver Economy Consultants

## Project Description
Interactive Streamlit app for identifying U.S. counties that balance:
- a strong **65+ population share**
- strong **household income / purchasing power**
- and, in Phase 2, better **affordability + access + lower risk**

## App Deployment URL

https://silvereconomyconsultants.streamlit.app

## Local Setup Instructions

```bash
git clone https://github.com/realkayembe/Silver-Economy-Consultants.git
cd Silver-Economy-Consultants
uv sync
uv run streamlit run app.py

### Or with the helper script
```bash
uv sync
./run.sh
```

### Project structure
The app is organized as a normal `uv` project:

- `pyproject.toml` — project metadata and dependencies
- `.python-version` — preferred Python version
- `app.py` — Streamlit application
- `data/` — packaged demo dataset
- `assets/` — screenshots and visual assets
- `AI_strategy.md` — AI feature design notes
- `run.sh` — convenience launcher

## Data modes inside the app
### 1) Packaged demo (default)
- Works **offline**.
- Phase 1 uses ACS-derived county-level data bundled with the project.
- Phase 2 uses packaged demo proxies so the app still works without internet.

### 2) Live ACS API
- Requires internet.
- Attempts to pull 2023 ACS 5-year profile variables directly from the Census API.
- Phase 1 variables:
  - `DP05_0024PE` — % age 65+
  - `DP03_0062E` — median household income
- Phase 2 variables:
  - `DP04_0089E` — median home value
  - `DP04_0134E` — median gross rent
  - `DP03_0099PE` — no health insurance coverage
  - `DP02_0078PE` — disability prevalence among 65+

## Scoring
### Phase 1
`Silver Score = w_age * z(% age 65+) + w_income * z(median household income)`

### Phase 2
`NASI = Silver Score + affordability adjustment + health/access adjustment + risk adjustment`

The app exposes the Phase 2 weights directly in the sidebar so you can tune the screener to the business strategy.

## Notes
- The packaged demo mode is designed to make the app runnable in classroom / offline settings.
- The live ACS mode is the intended production path.



