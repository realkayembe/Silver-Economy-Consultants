from __future__ import annotations
from pathlib import Path
import re
from io import StringIO
import pandas as pd
import numpy as np

# Compatibility shim for older xarray builds imported by plotly.express
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_

import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / 'data' / 'silver_economy_demo.csv'
ACS_YEAR = 2023

st.set_page_config(
    page_title='Silver Economy Market Screener',
    page_icon='',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    .block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
    .small-note { color: #6c7a92; font-size: 0.88rem; }
    .headline { color: #112542; }
    .insight-box {
        background: white; border: 1px solid #d8dee8; border-radius: 14px;
        padding: 1rem 1.1rem; box-shadow: 0 2px 10px rgba(17,37,66,.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Data helpers ----------
def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def safe_show_df(df: pd.DataFrame, *, use_container_width: bool = True, hide_index: bool = True):
    """Display a dataframe without hard-requiring PyArrow."""
    try:
        st.dataframe(df, use_container_width=use_container_width, hide_index=hide_index)
    except Exception:
        st.table(df if not hide_index else df.reset_index(drop=True))


@st.cache_data(show_spinner=False)
def load_demo_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, dtype={'fips': str})


@st.cache_data(show_spinner=False)
def fetch_phase1_from_acs(api_key: str | None = None) -> pd.DataFrame:
    vars_ = ['NAME', 'DP05_0024PE', 'DP03_0062E']
    params = {'get': ','.join(vars_), 'for': 'county:*'}
    if api_key:
        params['key'] = api_key
    url = f'https://api.census.gov/data/{ACS_YEAR}/acs/acs5/profile'
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    rows = r.json()
    hdr, vals = rows[0], rows[1:]
    df = pd.DataFrame(vals, columns=hdr)
    df['fips'] = df['state'].astype(str).str.zfill(2) + df['county'].astype(str).str.zfill(3)
    df['county'] = df['NAME'].str.split(',').str[0].str.replace(' County', '', regex=False)
    df['state_name'] = df['NAME'].str.split(',').str[-1].str.strip()
    out = pd.DataFrame({
        'state': df['state_name'],
        'county': df['county'],
        'fips': df['fips'],
        'age65andolder_pct': pd.to_numeric(df['DP05_0024PE'], errors='coerce'),
        'median_hh_inc': pd.to_numeric(df['DP03_0062E'], errors='coerce'),
    })
    return out.dropna()


@st.cache_data(show_spinner=False)
def fetch_phase2_from_acs(api_key: str | None = None) -> pd.DataFrame:
    vars_ = ['NAME', 'DP04_0089E', 'DP04_0134E', 'DP03_0099PE', 'DP02_0078PE']
    params = {'get': ','.join(vars_), 'for': 'county:*'}
    if api_key:
        params['key'] = api_key
    url = f'https://api.census.gov/data/{ACS_YEAR}/acs/acs5/profile'
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    rows = r.json()
    hdr, vals = rows[0], rows[1:]
    df = pd.DataFrame(vals, columns=hdr)
    df['fips'] = df['state'].astype(str).str.zfill(2) + df['county'].astype(str).str.zfill(3)
    out = pd.DataFrame({
        'fips': df['fips'],
        'median_home_value_usd': pd.to_numeric(df['DP04_0089E'], errors='coerce'),
        'median_rent_usd': pd.to_numeric(df['DP04_0134E'], errors='coerce'),
        'uninsured_pct': pd.to_numeric(df['DP03_0099PE'], errors='coerce'),
        'disability65_pct': pd.to_numeric(df['DP02_0078PE'], errors='coerce'),
    })
    return out


def build_offline_phase2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['median_home_value_usd'] = out['demo_home_value_usd']
    out['median_rent_usd'] = out['demo_median_rent_usd']
    out['rent_burden_pct'] = out['demo_rent_burden_pct']
    out['uninsured_pct'] = out['demo_uninsured_pct']
    out['disability65_pct'] = out['demo_disability65_pct']
    out['health_access_per_1k'] = out['demo_health_access_per_1k']
    out['risk_index'] = out['demo_risk_index']
    return out


def join_live_phase2(phase1: pd.DataFrame, demo_df: pd.DataFrame, api_key: str | None = None) -> tuple[pd.DataFrame, str]:
    phase2_live = fetch_phase2_from_acs(api_key)
    out = phase1.merge(phase2_live, on='fips', how='left')

    helper = demo_df.copy()
    if 'beds_per_1k' in helper.columns:
        helper['health_access_helper'] = pd.to_numeric(helper['beds_per_1k'], errors='coerce')
    elif 'demo_health_access_per_1k' in helper.columns:
        helper['health_access_helper'] = pd.to_numeric(helper['demo_health_access_per_1k'], errors='coerce')
    else:
        helper['health_access_helper'] = np.nan

    if 'demo_risk_index' in helper.columns:
        helper['risk_helper'] = pd.to_numeric(helper['demo_risk_index'], errors='coerce')
    elif 'risk_index' in helper.columns:
        helper['risk_helper'] = pd.to_numeric(helper['risk_index'], errors='coerce')
    else:
        helper['risk_helper'] = np.nan

    out = out.merge(helper[['fips', 'health_access_helper', 'risk_helper']], on='fips', how='left')
    out['rent_burden_pct'] = (12 * out['median_rent_usd'] / out['median_hh_inc'] * 100).replace([np.inf, -np.inf], np.nan)

    if 'health_access_per_1k' not in out.columns:
        out['health_access_per_1k'] = np.nan
    out['health_access_per_1k'] = out['health_access_per_1k'].fillna(out['health_access_helper'])

    if 'risk_index' not in out.columns:
        out['risk_index'] = np.nan
    out['risk_index'] = out['risk_index'].fillna(out['risk_helper'])

    out['health_access_per_1k'] = out['health_access_per_1k'].fillna(out['health_access_per_1k'].median())
    out['risk_index'] = out['risk_index'].fillna(out['risk_index'].median())

    for col in ['median_home_value_usd', 'median_rent_usd', 'uninsured_pct', 'disability65_pct', 'rent_burden_pct']:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors='coerce')
        out[col] = out[col].fillna(out[col].median())

    return out, 'Live ACS 2023 loaded. Core ACS fields come from the Census API; unavailable non-ACS helper fields are filled with packaged fallbacks or neutral defaults so live mode stays on.'


def compute_scores(df: pd.DataFrame, w_age: float, w_income: float,
                   w_afford: float, w_health: float, w_risk: float) -> pd.DataFrame:
    out = df.copy()
    out['z_age65'] = zscore(out['age65andolder_pct'])
    out['z_income'] = zscore(out['median_hh_inc'])
    out['silver_score'] = w_age * out['z_age65'] + w_income * out['z_income']

    out['home_to_income_ratio'] = out['median_home_value_usd'] / out['median_hh_inc']
    out['z_rent_burden'] = zscore(out['rent_burden_pct'])
    out['z_home_ratio'] = zscore(out['home_to_income_ratio'])
    out['z_uninsured'] = zscore(out['uninsured_pct'])
    out['z_disability65'] = zscore(out['disability65_pct'])
    out['z_health_access'] = zscore(out['health_access_per_1k'])
    out['z_risk'] = zscore(out['risk_index'])

    out['affordability_component'] = -(0.6 * out['z_rent_burden'] + 0.4 * out['z_home_ratio'])
    out['health_component'] = (-0.55 * out['z_uninsured'] - 0.25 * out['z_disability65'] + 0.20 * out['z_health_access'])
    out['risk_component'] = -out['z_risk']

    out['affordability_adj'] = w_afford * out['affordability_component']
    out['health_adj'] = w_health * out['health_component']
    out['risk_adj'] = w_risk * out['risk_component']

    out['nasi_score'] = out['silver_score'] + out['affordability_adj'] + out['health_adj'] + out['risk_adj']
    out['phase1_rank'] = out['silver_score'].rank(ascending=False, method='min')
    out['phase2_rank'] = out['nasi_score'].rank(ascending=False, method='min')
    out['rank_delta'] = out['phase1_rank'] - out['phase2_rank']
    return out


# ---------- Visuals ----------
def make_phase1_scatter(df: pd.DataFrame):
    hover_cols = {'state': True, 'county_population': ':,', 'silver_score': ':.2f'}
    fig = px.scatter(
        df,
        x='age65andolder_pct',
        y='median_hh_inc',
        color='silver_score',
        size='county_population' if 'county_population' in df.columns else None,
        size_max=24,
        hover_name='county',
        hover_data=hover_cols,
        labels={'age65andolder_pct': '% age 65+', 'median_hh_inc': 'Median household income ($)', 'silver_score': 'Silver Score'},
    )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=42, b=10), title='Phase 1 — County opportunity space')
    return fig


def make_rank_bar(df: pd.DataFrame, score_col: str, title: str, top_n: int = 15):
    d = df.sort_values(score_col, ascending=False).head(top_n).copy()
    d['label'] = d['county'] + ', ' + d['state']
    fig = px.bar(d.iloc[::-1], x=score_col, y='label', orientation='h', text=score_col)
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside', cliponaxis=False)
    fig.update_layout(height=460, margin=dict(l=10, r=50, t=42, b=10), title=title, yaxis_title='')
    return fig


def make_rank_shift(df: pd.DataFrame, top_n: int = 15):
    d = df.sort_values('nasi_score', ascending=False).head(top_n).copy()
    d['label'] = d['county'] + ', ' + d['state']
    d = d.sort_values('phase2_rank', ascending=True)
    fig = go.Figure()
    for _, r in d.iterrows():
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[r['phase1_rank'], r['phase2_rank']],
            mode='lines+markers+text',
            text=[r['label'], ''],
            textposition='middle left',
            hovertemplate=f"{r['label']}<br>Phase 1 rank: {int(r['phase1_rank'])}<br>Phase 2 rank: {int(r['phase2_rank'])}<br>Rank delta: {int(r['rank_delta'])}<extra></extra>",
            showlegend=False
        ))
    fig.update_layout(
        title='Phase 2 — who moves up or down after affordability + health adjustment',
        height=460,
        margin=dict(l=10, r=10, t=42, b=10),
        xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Phase 1 rank', 'Phase 2 rank'], range=[-0.1, 1.2]),
        yaxis=dict(autorange='reversed', title='Rank (1 = best)')
    )
    return fig


def make_score_compare(df: pd.DataFrame):
    fig = px.scatter(
        df,
        x='silver_score',
        y='nasi_score',
        color='rank_delta',
        hover_name='county',
        hover_data={'state': True, 'rank_delta': True, 'silver_score': ':.2f', 'nasi_score': ':.2f'},
        labels={'silver_score': 'Phase 1 Silver Score', 'nasi_score': 'Phase 2 NASI'}
    )
    mn = float(min(df['silver_score'].min(), df['nasi_score'].min()))
    mx = float(max(df['silver_score'].max(), df['nasi_score'].max()))
    fig.add_shape(type='line', x0=mn, y0=mn, x1=mx, y1=mx, line=dict(dash='dash'))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=42, b=10), title='Before vs. after Phase 2 adjustment')
    return fig


def county_card(df: pd.DataFrame, mode_note: str):
    if df.empty:
        return
    ordered = df.sort_values('nasi_score', ascending=False)
    options = (ordered['county'] + ', ' + ordered['state']).tolist()
    choice = st.selectbox('County profile', options, index=0, key='county_profile_select')
    county, state = choice.split(', ', 1)
    r = df[(df['county'] == county) & (df['state'] == state)].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Phase 1 Silver Score', f"{r['silver_score']:.2f}")
    c2.metric('Phase 2 NASI', f"{r['nasi_score']:.2f}", delta=f"{r['nasi_score'] - r['silver_score']:.2f}")
    c3.metric('Median income', f"${r['median_hh_inc']:,.0f}")
    c4.metric('% age 65+', f"{r['age65andolder_pct']:.1f}%")

    dcomp = pd.DataFrame({
        'component': ['Silver base', 'Affordability adj.', 'Health/access adj.', 'Risk adj.'],
        'value': [r['silver_score'], r['affordability_adj'], r['health_adj'], r['risk_adj']]
    })
    fig = px.bar(dcomp, x='component', y='value', text='value', title='Score decomposition (weighted components used in NASI)')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=42, b=10), xaxis_title='')
    st.plotly_chart(fig, use_container_width=True)
    st.caption(mode_note)


# ---------- AI helpers ----------
def label_for_row(r: pd.Series) -> str:
    return f"{r['county']}, {r['state']}"


def county_tier(value: float) -> str:
    if value >= 1.5:
        return 'standout'
    if value >= 0.5:
        return 'strong'
    if value >= -0.5:
        return 'middle-of-the-pack'
    return 'weak'


def row_from_label(df: pd.DataFrame, label: str) -> pd.Series:
    county, state = label.split(', ', 1)
    return df[(df['county'] == county) & (df['state'] == state)].iloc[0]


def build_market_brief(r: pd.Series, mode_note: str) -> str:
    strengths = []
    watchouts = []
    if r['z_age65'] > 0.5:
        strengths.append(f"older-than-average population ({r['age65andolder_pct']:.1f}% age 65+)")
    elif r['z_age65'] < -0.5:
        watchouts.append(f"below-average senior concentration ({r['age65andolder_pct']:.1f}% age 65+)")

    if r['z_income'] > 0.5:
        strengths.append(f"strong household income (${r['median_hh_inc']:,.0f})")
    elif r['z_income'] < -0.5:
        watchouts.append(f"weaker purchasing power (${r['median_hh_inc']:,.0f})")

    if r['affordability_adj'] > 0.1:
        strengths.append('affordability is helping rather than hurting the market')
    elif r['affordability_adj'] < -0.1:
        watchouts.append('affordability pressure is dragging the score down')

    if r['health_adj'] > 0.1:
        strengths.append('health/access conditions are supportive')
    elif r['health_adj'] < -0.1:
        watchouts.append('health/access indicators are weaker than the peer set')

    if r['risk_adj'] > 0.1:
        strengths.append('risk conditions are better than average')
    elif r['risk_adj'] < -0.1:
        watchouts.append('risk exposure is worse than average')

    strengths_text = '; '.join(strengths[:3]) if strengths else 'no major strengths clearly separate this county from the current peer set'
    watchouts_text = '; '.join(watchouts[:3]) if watchouts else 'no major red flags stand out in the current filter set'

    return (
        f"**AI market brief — {label_for_row(r)}**\n\n"
        f"This county looks **{county_tier(r['nasi_score'])}** after the full Phase 2 screen. "
        f"Its Phase 1 Silver Score is **{r['silver_score']:.2f}** and its Phase 2 NASI is **{r['nasi_score']:.2f}**, "
        f"which means the second-stage adjustments changed the case by **{r['nasi_score'] - r['silver_score']:+.2f}** points.\n\n"
        f"**What is helping:** {strengths_text}.\n\n"
        f"**What to watch:** {watchouts_text}.\n\n"
        f"**Recommendation:** treat this county as a {'high-priority finalist' if r['nasi_score'] >= 1 else 'conditional candidate' if r['nasi_score'] >= 0 else 'lower-priority market'} and validate tract-level demand, competitor presence, zoning, and land availability next.\n\n"
        f"_Data mode note: {mode_note}_"
    )


def compare_counties_brief(r1: pd.Series, r2: pd.Series) -> str:
    winner = r1 if r1['nasi_score'] >= r2['nasi_score'] else r2
    loser = r2 if winner is r1 else r1
    lines = []
    for metric, label, higher_better in [
        ('age65andolder_pct', '% age 65+', True),
        ('median_hh_inc', 'median income', True),
        ('affordability_adj', 'affordability adjustment', True),
        ('health_adj', 'health/access adjustment', True),
        ('risk_adj', 'risk adjustment', True),
    ]:
        if winner[metric] == loser[metric]:
            continue
        better = winner if (winner[metric] > loser[metric]) == higher_better else loser
        lines.append(f"- **{label}** favors **{label_for_row(better)}**.")

    return (
        f"**AI county comparison**\n\n"
        f"Between **{label_for_row(r1)}** and **{label_for_row(r2)}**, the stronger overall market in the current scenario is "
        f"**{label_for_row(winner)}** with NASI **{winner['nasi_score']:.2f}** versus **{loser['nasi_score']:.2f}**.\n\n"
        + '\n'.join(lines[:5])
    )


def data_quality_brief(df: pd.DataFrame, data_mode_label: str, mode_note: str) -> str:
    keys = ['age65andolder_pct', 'median_hh_inc', 'median_home_value_usd', 'median_rent_usd', 'uninsured_pct', 'disability65_pct', 'health_access_per_1k', 'risk_index']
    missing = {k: int(df[k].isna().sum()) for k in keys if k in df.columns}
    missing_text = ', '.join([f"{k}: {v}" for k, v in missing.items() if v > 0]) or 'none of the tracked fields are currently missing after scoring input preparation'
    return (
        f"**AI data-quality explainer**\n\n"
        f"Current mode: **{data_mode_label}**. {mode_note}\n\n"
        f"Missing-value check across key fields: {missing_text}.\n\n"
        f"Interpretation: if you are in packaged demo mode, Phase 2 uses bundled proxy fields so the app stays runnable offline. "
        f"If you are in live ACS mode, rent, home value, uninsured share, and disability share come from ACS, while health-access and risk still use helper proxies bundled with the app."
    )


def parse_ai_query(query: str, states_available: list[str]) -> dict:
    q = (query or '').strip().lower()
    prefs = {
        'top_n': 10,
        'states': [],
        'emphasis': [],
        'sort_col': 'nasi_score',
    }
    m = re.search(r'\btop\s+(\d{1,2})\b', q)
    if m:
        prefs['top_n'] = max(3, min(25, int(m.group(1))))

    for state in states_available:
        if state.lower() in q:
            prefs['states'].append(state)

    keyword_map = {
        'affordable': 'affordability_adj',
        'less affordable': 'affordability_adj',
        'low risk': 'risk_adj',
        'safer': 'risk_adj',
        'health access': 'health_adj',
        'healthcare': 'health_adj',
        'income': 'median_hh_inc',
        'wealth': 'median_hh_inc',
        'older': 'age65andolder_pct',
        'senior': 'age65andolder_pct',
        'population': 'county_population',
        'rural': 'rural_pct',
        'dense': 'density_pop',
    }
    for key, val in keyword_map.items():
        if key in q:
            prefs['emphasis'].append(val)

    if 'phase 1' in q or 'silver score' in q:
        prefs['sort_col'] = 'silver_score'
    return prefs


def run_ai_scenario(df: pd.DataFrame, query: str, states_available: list[str]) -> tuple[pd.DataFrame, str]:
    prefs = parse_ai_query(query, states_available)
    out = df.copy()
    if prefs['states']:
        out = out[out['state'].isin(prefs['states'])]

    out = out.copy()
    out['ai_score'] = out[prefs['sort_col']]
    explanation_bits = [f"base sort = {prefs['sort_col']}"]

    for col in prefs['emphasis']:
        if col in out.columns:
            if col == 'rural_pct':
                out['ai_score'] += 0.30 * zscore(out[col])
                explanation_bits.append('preferred more rural markets')
            elif col == 'density_pop':
                out['ai_score'] += 0.30 * zscore(out[col])
                explanation_bits.append('preferred denser markets')
            else:
                out['ai_score'] += 0.45 * zscore(out[col])
                explanation_bits.append(f'emphasized {col}')

    out = out.sort_values('ai_score', ascending=False).head(prefs['top_n'])
    explanation = 'AI scenario parser: ' + '; '.join(explanation_bits)
    if prefs['states']:
        explanation += f"; state filter inferred = {', '.join(prefs['states'])}"
    return out, explanation


def scenario_snapshot(name: str, states: list[str], min_age: float, min_inc: int, min_pop: int,
                      w_age: float, w_income: float, w_afford: float, w_health: float, w_risk: float,
                      top_county: str) -> dict:
    return {
        'name': name,
        'states': ', '.join(states) if states else 'All states',
        'min_age': min_age,
        'min_inc': min_inc,
        'min_pop': min_pop,
        'w_age': round(w_age, 2),
        'w_income': round(w_income, 2),
        'w_afford': round(w_afford, 2),
        'w_health': round(w_health, 2),
        'w_risk': round(w_risk, 2),
        'top_county': top_county,
    }


# ---------- UI ----------
st.markdown('<h1 class="headline">🏡 Silver Economy Market Screener</h1>', unsafe_allow_html=True)
st.caption('Interactive market-screening app for affluent retirement community siting. Phase 1 identifies wealthy-senior markets; Phase 2 adjusts for affordability, access, and risk.')

demo_df = load_demo_data()
states_available = sorted(demo_df['state'].dropna().unique().tolist())
if 'saved_scenarios' not in st.session_state:
    st.session_state.saved_scenarios = []

with st.sidebar:
    st.header('Controls')
    data_mode = st.radio('Data source', ['Packaged demo (offline)', 'Live ACS API (internet required)'], index=0)
    api_key = ''
    if data_mode.startswith('Live ACS'):
        api_key = st.text_input('Optional Census API key', type='password', help='Not required for light use, but recommended.')
    st.divider()

    states = st.multiselect('State filter', states_available, default=[])
    min_age = st.slider('Min % age 65+', 0.0, 40.0, 12.0, 0.5)
    min_inc = st.slider('Min median household income', 0, 150000, 50000, 2500)
    min_pop = st.slider('Min county population', 0, 1500000, 25000, 5000)
    top_n = st.slider('Top N to display', 5, 25, 15, 1)

    st.divider()
    st.subheader('Phase 1 weights')
    w_age = st.slider('Weight: % age 65+', 0.0, 1.0, 0.50, 0.05)
    w_income = 1.0 - w_age
    st.caption(f'Income weight auto-balances to {w_income:.2f}')

    st.subheader('Phase 2 adjustment weights')
    w_afford = st.slider('Affordability weight', 0.0, 1.0, 0.35, 0.05)
    w_health = st.slider('Health/access weight', 0.0, 1.0, 0.25, 0.05)
    w_risk = st.slider('Risk weight', 0.0, 1.0, 0.20, 0.05)

mode_note = ''
if data_mode.startswith('Live ACS'):
    try:
        phase1_live = fetch_phase1_from_acs(api_key or None)
        merged = phase1_live.merge(
            demo_df[
                ['fips', 'county_population', 'density_pop', 'rural_pct', 'demo_home_value_usd', 'demo_median_rent_usd',
                 'demo_rent_burden_pct', 'demo_uninsured_pct', 'demo_disability65_pct', 'demo_health_access_per_1k',
                 'demo_risk_index'] + (['beds_per_1k'] if 'beds_per_1k' in demo_df.columns else [])
            ],
            on='fips', how='left'
        )
        merged['state'] = merged['state'].fillna('Unknown')
        merged['county_population'] = merged['county_population'].fillna(0)
        merged['density_pop'] = merged['density_pop'].fillna(0)
        merged['rural_pct'] = merged['rural_pct'].fillna(0)
        merged, mode_note = join_live_phase2(merged, demo_df, api_key or None)
        data = merged
    except Exception as e:
        st.sidebar.warning(f'Live ACS pull failed, so the app fell back to packaged demo data. Details: {e}')
        data = build_offline_phase2(demo_df)
        mode_note = 'Packaged demo mode: Phase 1 uses ACS-derived county data and Phase 2 uses bundled demo proxies so the app runs offline.'
else:
    data = build_offline_phase2(demo_df)
    mode_note = 'Packaged demo mode: Phase 1 uses ACS-derived county data and Phase 2 uses bundled demo proxies so the app runs offline.'

data = compute_scores(data, w_age=w_age, w_income=w_income, w_afford=w_afford, w_health=w_health, w_risk=w_risk)
filtered = data.copy()
if states:
    filtered = filtered[filtered['state'].isin(states)]
filtered = filtered[filtered['age65andolder_pct'] >= min_age]
filtered = filtered[filtered['median_hh_inc'] >= min_inc]
filtered = filtered[filtered['county_population'].fillna(0) >= min_pop]
filtered = filtered.sort_values('silver_score', ascending=False)

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric('Counties in filter', f'{len(filtered):,}')
mc2.metric('Median % 65+', f"{filtered['age65andolder_pct'].median():.1f}%" if len(filtered) else '—')
mc3.metric('Median income', f"${filtered['median_hh_inc'].median():,.0f}" if len(filtered) else '—')
mc4.metric('Best Phase 2 market', f"{filtered.sort_values('nasi_score', ascending=False).iloc[0]['county']}, {filtered.sort_values('nasi_score', ascending=False).iloc[0]['state']}" if len(filtered) else '—')
st.caption(mode_note)

tab1, tab2, tab3 = st.tabs(['Phase 1: MVP demo', 'Phase 2: NASI demo', 'AI Copilot'])

with tab1:
    st.subheader('Find counties with both seniors and purchasing power')
    c1, c2 = st.columns([1.4, 1.0])
    with c1:
        st.plotly_chart(make_phase1_scatter(filtered), use_container_width=True)
    with c2:
        st.plotly_chart(make_rank_bar(filtered, 'silver_score', 'Top counties by Phase 1 Silver Score', top_n=top_n), use_container_width=True)

    shortlist = filtered.sort_values('silver_score', ascending=False).head(top_n).copy()
    shortlist['county_label'] = shortlist['county'] + ', ' + shortlist['state']
    view_cols = ['county_label', 'age65andolder_pct', 'median_hh_inc', 'silver_score', 'county_population']
    safe_show_df(shortlist[view_cols].rename(columns={
        'county_label': 'County', 'age65andolder_pct': '% age 65+', 'median_hh_inc': 'Median income', 'silver_score': 'Silver Score', 'county_population': 'Population'
    }), use_container_width=True, hide_index=True)
    st.download_button('Download Phase 1 shortlist CSV', shortlist.to_csv(index=False).encode('utf-8'), file_name='phase1_shortlist.csv', mime='text/csv')

with tab2:
    st.subheader('Adjust the shortlist for affordability, access, and risk')
    c1, c2 = st.columns([1.1, 1.1])
    with c1:
        st.plotly_chart(make_score_compare(filtered), use_container_width=True)
    with c2:
        st.plotly_chart(make_rank_shift(filtered, top_n=top_n), use_container_width=True)

    c3, c4 = st.columns([1.0, 1.0])
    with c3:
        st.plotly_chart(make_rank_bar(filtered, 'nasi_score', 'Top counties by Phase 2 NASI', top_n=top_n), use_container_width=True)
    with c4:
        movers = filtered[['county', 'state', 'phase1_rank', 'phase2_rank', 'rank_delta']].copy()
        movers = movers.sort_values('rank_delta', ascending=False).head(top_n)
        movers['County'] = movers['county'] + ', ' + movers['state']
        safe_show_df(movers[['County', 'phase1_rank', 'phase2_rank', 'rank_delta']].rename(columns={
            'phase1_rank': 'Phase 1 rank', 'phase2_rank': 'Phase 2 rank', 'rank_delta': 'Improvement'
        }), use_container_width=True, hide_index=True)

    st.markdown('#### County deep dive')
    county_card(filtered, mode_note)

with tab3:
    st.subheader('AI Copilot — now implemented inside the app')
    st.caption('This version uses deterministic local logic instead of an external LLM, so it runs today without another API key.')

    ai1, ai2, ai3 = st.tabs(['Market brief', 'Scenario assistant', 'Compare & remember'])

    with ai1:
        if filtered.empty:
            st.warning('No counties match the current filters.')
        else:
            labels = (filtered.sort_values('nasi_score', ascending=False).assign(label=lambda d: d['county'] + ', ' + d['state'])['label'].tolist())
            selected_label = st.selectbox('Pick a county for an AI-generated brief', labels, index=0, key='ai_brief_county')
            brief_row = row_from_label(filtered, selected_label)
            brief = build_market_brief(brief_row, mode_note)
            st.markdown(brief)
            st.download_button('Download county brief (.md)', brief.encode('utf-8'), file_name=f"{selected_label.replace(', ', '_').replace(' ', '_').lower()}_brief.md", mime='text/markdown')

            st.markdown('#### Data-quality explainer')
            st.markdown(data_quality_brief(filtered, data_mode, mode_note))

    with ai2:
        prompt = st.text_input(
            'Ask the copilot for a scenario',
            value='Top 10 affordable, lower-risk senior markets in Virginia',
            help='Examples: “Top 8 senior markets with strong income”, “Top 12 lower-risk counties in Florida”, “Top 10 Phase 1 markets in Texas”.'
        )
        scenario_df, scenario_explainer = run_ai_scenario(filtered if not filtered.empty else data, prompt, states_available)
        st.info(scenario_explainer)
        if scenario_df.empty:
            st.warning('The AI scenario returned no counties with the current base filter set.')
        else:
            display = scenario_df[['county', 'state', 'silver_score', 'nasi_score', 'ai_score', 'age65andolder_pct', 'median_hh_inc']].copy()
            display.columns = ['County', 'State', 'Silver Score', 'NASI', 'AI Score', '% age 65+', 'Median income']
            safe_show_df(display.reset_index(drop=True), use_container_width=True, hide_index=True)
            fig = make_rank_bar(scenario_df.rename(columns={'ai_score': 'ai_score'}), 'ai_score', 'AI scenario recommendations', top_n=min(15, len(scenario_df)))
            st.plotly_chart(fig, use_container_width=True)
            top_label = label_for_row(scenario_df.iloc[0])
            st.success(f'AI pick for this prompt: {top_label}')

    with ai3:
        left, right = st.columns(2)
        labels = (filtered.sort_values('nasi_score', ascending=False).assign(label=lambda d: d['county'] + ', ' + d['state'])['label'].tolist()) if len(filtered) else []
        with left:
            c_a = st.selectbox('County A', labels, index=0 if labels else None, key='compare_a') if labels else None
        with right:
            c_b = st.selectbox('County B', labels, index=1 if len(labels) > 1 else 0 if labels else None, key='compare_b') if labels else None
        if c_a and c_b:
            r1 = row_from_label(filtered, c_a)
            r2 = row_from_label(filtered, c_b)
            st.markdown(compare_counties_brief(r1, r2))

        st.markdown('#### Recommendation memory')
        save_name = st.text_input('Name this current scenario', value='Retirement screener scenario')
        top_county = f"{filtered.sort_values('nasi_score', ascending=False).iloc[0]['county']}, {filtered.sort_values('nasi_score', ascending=False).iloc[0]['state']}" if len(filtered) else '—'
        if st.button('Save current scenario'):
            st.session_state.saved_scenarios.append(
                scenario_snapshot(save_name, states, min_age, min_inc, min_pop, w_age, w_income, w_afford, w_health, w_risk, top_county)
            )
            st.success('Scenario saved for this session.')
        if st.session_state.saved_scenarios:
            saved_df = pd.DataFrame(st.session_state.saved_scenarios)
            safe_show_df(saved_df, use_container_width=True, hide_index=True)
            csv_bytes = saved_df.to_csv(index=False).encode('utf-8')
            st.download_button('Download saved scenarios CSV', csv_bytes, file_name='saved_scenarios.csv', mime='text/csv')
        else:
            st.caption('No saved scenarios yet.')

st.markdown('---')
st.caption('Phase 1 formula: Silver Score = weight_age · z(% age 65+) + weight_income · z(median household income).')
st.caption('Phase 2 formula: NASI = Silver Score + affordability adjustment + health/access adjustment + risk adjustment.')
st.caption('AI Copilot implementation in this build: local market-brief generator, scenario parser, county comparison assistant, data-quality explainer, and session memory. These features are fully coded without relying on an external LLM API.')
