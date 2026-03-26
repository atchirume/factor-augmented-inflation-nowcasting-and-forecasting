#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Factor-Augmented Inflation Framework",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# THEME
# =========================================================
st.markdown(
    """
<style>
:root {
    --rbz-navy: #0B1F3A;
    --rbz-blue: #174EA6;
    --rbz-cyan: #1F6FB2;
    --rbz-gold: #D4A017;
    --rbz-bg: #F6F9FC;
    --rbz-card: rgba(255,255,255,0.96);
    --rbz-text: #132238;
    --rbz-muted: #6B7A90;
    --rbz-border: rgba(23,78,166,0.12);
    --rbz-shadow: 0 14px 34px rgba(11,31,58,0.10);
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(23,78,166,0.12), transparent 24%),
        radial-gradient(circle at top right, rgba(212,160,23,0.11), transparent 18%),
        linear-gradient(180deg, #F7FAFD 0%, #EEF4FA 52%, #F8FBFE 100%);
    color: var(--rbz-text);
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.5rem;
    max-width: 1500px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--rbz-navy) 0%, #102B4C 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F5F8FC !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: #132238 !important;
    border-radius: 12px !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] span,
[data-testid="stSidebar"] div[data-baseweb="select"] input {
    color: #132238 !important;
    -webkit-text-fill-color: #132238 !important;
}

.rbz-hero {
    background: linear-gradient(135deg, #0B1F3A 0%, #174EA6 58%, #D4A017 140%);
    padding: 28px 30px;
    border-radius: 24px;
    color: white;
    box-shadow: 0 20px 44px rgba(11,31,58,0.22);
    margin-bottom: 18px;
}
.rbz-hero h1 {
    margin: 0 0 6px 0;
    font-size: 2.05rem;
    line-height: 1.1;
    font-weight: 800;
}
.rbz-hero p {
    margin: 0;
    font-size: 1rem;
    color: rgba(255,255,255,0.92);
}
.rbz-subline {
    margin-top: 10px;
    font-size: 0.92rem;
    color: rgba(255,255,255,0.88);
}

.metric-card {
    background: var(--rbz-card);
    border: 1px solid var(--rbz-border);
    border-radius: 22px;
    padding: 16px 18px;
    box-shadow: var(--rbz-shadow);
    min-height: 122px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 6px;
    background: linear-gradient(90deg, var(--rbz-blue), var(--rbz-gold));
}
.metric-title {
    font-size: 13px;
    color: var(--rbz-muted);
    margin-bottom: 8px;
    font-weight: 600;
    text-transform: uppercase;
}
.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: var(--rbz-navy);
    line-height: 1.08;
}
.metric-note {
    font-size: 12px;
    color: var(--rbz-muted);
    margin-top: 8px;
}

.policy-card {
    background: linear-gradient(135deg, rgba(11,31,58,0.98) 0%, rgba(23,78,166,0.96) 100%);
    color: white;
    border-radius: 22px;
    padding: 20px 22px;
    box-shadow: 0 18px 38px rgba(11,31,58,0.18);
    margin: 10px 0 18px 0;
}
.policy-card h3 {
    margin: 0 0 10px 0;
    font-size: 1.2rem;
    color: #fff;
}
.policy-card p {
    margin: 4px 0;
    color: rgba(255,255,255,0.92);
}

div[data-testid="stDataFrame"],
div[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,0.92);
    border-radius: 18px;
    padding: 6px 8px;
    box-shadow: 0 10px 24px rgba(11,31,58,0.06);
    border: 1px solid rgba(23,78,166,0.08);
}

button[kind="primary"], .stDownloadButton button {
    border-radius: 14px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# GLOBAL SETTINGS
# =========================================================
DEFAULT_DATA_PATH = "simulated_zimbabwe_inflation_data_augmented.csv"

SADC_LOWER = 3.0
SADC_UPPER = 7.0

TARGET_OPTIONS = {
    "Monthly Inflation": "monthly_inflation_from_cpi",
    "Annual Inflation": "annual_inflation_from_cpi",
    "Quarterly Inflation (QoQ)": "quarterly_inflation_qoq",
    "Quarterly Inflation (Annualised)": "quarterly_inflation_annualised",
    "Core Monthly Inflation": "core_monthly_inflation",
    "Core Annual Inflation": "core_annual_inflation",
    "Core Quarterly Inflation (QoQ)": "core_quarterly_inflation_qoq",
    "Core Quarterly Inflation (Annualised)": "core_quarterly_inflation_annualised",
}

BASE_DRIVER_COLS = [
    "exrate_change",
    "fuel_change",
    "m3_growth",
    "reserve_money_growth",
    "rtgs_growth",
    "mobile_growth",
    "pos_growth",
    "ppi_change",
    "cempi_change",
    "pdl_change",
    "policy_rate",
    "credit_private_sector_growth",
    "wage_growth_proxy",
    "output_gap_proxy",
    "fiscal_impulse_proxy",
    "inflation_expectations_proxy",
    "imported_inflation_proxy",
    "exchange_rate_volatility_3m",
    "inflation_diffusion_proxy",
]

FACTOR_BLOCKS = {
    "domestic_liquidity_factor": [
        "m3_growth",
        "reserve_money_growth",
        "rtgs_growth",
        "mobile_growth",
        "pos_growth",
        "credit_private_sector_growth",
        "policy_rate",
    ],
    "external_cost_factor": [
        "exrate_change",
        "exchange_rate_volatility_3m",
        "fuel_change",
        "imported_inflation_proxy",
        "cempi_change",
        "ppi_change",
    ],
    "inflation_momentum_factor": [
        "core_monthly_inflation",
        "core_annual_inflation",
        "inflation_expectations_proxy",
        "inflation_diffusion_proxy",
        "wage_growth_proxy",
    ],
    "demand_fiscal_factor": [
        "output_gap_proxy",
        "fiscal_impulse_proxy",
        "pdl_change",
    ],
}

FACTOR_NAMES = list(FACTOR_BLOCKS.keys())

REGIME_LABELS = [
    "Below Lower Band",
    "Within SADC Band",
    "Moderately Above Band",
    "Far Above Band",
]
REGIME_MAP = {label: i for i, label in enumerate(REGIME_LABELS)}
REVERSE_REGIME_MAP = {i: label for label, i in REGIME_MAP.items()}

SCENARIO_LIBRARY = {
    "Neutral": {
        "policy_rate": 0.0,
        "m3_growth": 0.0,
        "reserve_money_growth": 0.0,
        "exrate_change": 0.0,
        "fuel_change": 0.0,
        "fiscal_impulse_proxy": 0.0,
        "inflation_expectations_proxy": 0.0,
    },
    "Hawkish": {
        "policy_rate": 3.0,
        "m3_growth": -2.0,
        "reserve_money_growth": -2.0,
        "exrate_change": -1.0,
        "fuel_change": 0.0,
        "fiscal_impulse_proxy": -1.0,
        "inflation_expectations_proxy": -1.5,
    },
    "Dovish": {
        "policy_rate": -2.0,
        "m3_growth": 2.5,
        "reserve_money_growth": 2.0,
        "exrate_change": 1.0,
        "fuel_change": 0.0,
        "fiscal_impulse_proxy": 1.0,
        "inflation_expectations_proxy": 1.5,
    },
    "Exchange-Rate Shock": {
        "policy_rate": 0.0,
        "m3_growth": 0.0,
        "reserve_money_growth": 0.0,
        "exrate_change": 4.0,
        "fuel_change": 1.0,
        "fiscal_impulse_proxy": 0.0,
        "inflation_expectations_proxy": 2.0,
    },
    "Disinflation Support": {
        "policy_rate": 1.0,
        "m3_growth": -1.5,
        "reserve_money_growth": -1.0,
        "exrate_change": -0.8,
        "fuel_change": -0.5,
        "fiscal_impulse_proxy": -0.5,
        "inflation_expectations_proxy": -1.2,
    },
}


# =========================================================
# DATACLASSES
# =========================================================
@dataclass
class ForecastBundle:
    model_df: pd.DataFrame
    metrics: pd.DataFrame
    backtest: pd.DataFrame
    best_model_name: str
    predictor_cols: List[str]
    fitted_models: Dict[str, object]
    residual_std: float
    contribution_table: pd.DataFrame


@dataclass
class RegimeBundle:
    class_results: pd.DataFrame
    confusion: pd.DataFrame
    report: pd.DataFrame
    accuracy: float


# =========================================================
# UI HELPERS
# =========================================================
def metric_card(title: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# DATA HELPERS
# =========================================================
def parse_date_column(date_series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(date_series, format="%d/%m/%y", errors="coerce")
    missing_mask = parsed.isna()
    if missing_mask.any():
        parsed.loc[missing_mask] = pd.to_datetime(
            date_series.loc[missing_mask],
            errors="coerce",
            dayfirst=True,
        )
    return parsed


def sanitize_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
        if out[col].isna().all():
            out[col] = 0.0
        else:
            out[col] = out[col].interpolate(method="linear", limit_direction="both")
            out[col] = out[col].fillna(out[col].median())
            out[col] = out[col].fillna(0.0)
    return out


@st.cache_data(show_spinner=False)
def safe_read_csv(file_or_path) -> pd.DataFrame:
    df = pd.read_csv(file_or_path)
    df.columns = df.columns.str.strip()

    if "date" not in df.columns:
        raise ValueError("The dataset must contain a 'date' column.")
    if "cpi" not in df.columns:
        raise ValueError("The dataset must contain a 'cpi' column.")

    df["date"] = parse_date_column(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    numeric_cols = [c for c in df.columns if c != "date"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def derive_cpi_measures(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "monthly_inflation_from_cpi" not in out.columns:
        out["monthly_inflation_from_cpi"] = (out["cpi"] / out["cpi"].shift(1) - 1.0) * 100.0

    if "annual_inflation_from_cpi" not in out.columns:
        out["annual_inflation_from_cpi"] = (out["cpi"] / out["cpi"].shift(12) - 1.0) * 100.0

    if "quarterly_inflation_qoq" not in out.columns:
        out["quarterly_inflation_qoq"] = (out["cpi"] / out["cpi"].shift(3) - 1.0) * 100.0

    if "quarterly_inflation_annualised" not in out.columns:
        out["quarterly_inflation_annualised"] = ((out["cpi"] / out["cpi"].shift(3)) ** 4 - 1.0) * 100.0

    if "monthly_inflation_3mma" not in out.columns:
        out["monthly_inflation_3mma"] = out["monthly_inflation_from_cpi"].rolling(3).mean()

    if "monthly_inflation_6mma" not in out.columns:
        out["monthly_inflation_6mma"] = out["monthly_inflation_from_cpi"].rolling(6).mean()

    return out


def ensure_core_series(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "core_cpi" not in out.columns:
        nonfood = out["nonfood_cpi"] if "nonfood_cpi" in out.columns else out["cpi"] * 0.98
        nontrad = out["nontradables_cpi"] if "nontradables_cpi" in out.columns else out["cpi"] * 0.99
        trad = out["tradables_cpi"] if "tradables_cpi" in out.columns else out["cpi"] * 1.00
        admin = out["administered_price_cpi"] if "administered_price_cpi" in out.columns else out["cpi"] * 1.01
        out["core_cpi"] = 0.35 * nonfood + 0.30 * nontrad + 0.20 * trad + 0.15 * admin

    if "core_monthly_inflation" not in out.columns:
        out["core_monthly_inflation"] = (out["core_cpi"] / out["core_cpi"].shift(1) - 1.0) * 100.0

    if "core_annual_inflation" not in out.columns:
        out["core_annual_inflation"] = (out["core_cpi"] / out["core_cpi"].shift(12) - 1.0) * 100.0

    if "core_quarterly_inflation_qoq" not in out.columns:
        out["core_quarterly_inflation_qoq"] = (out["core_cpi"] / out["core_cpi"].shift(3) - 1.0) * 100.0

    if "core_quarterly_inflation_annualised" not in out.columns:
        out["core_quarterly_inflation_annualised"] = ((out["core_cpi"] / out["core_cpi"].shift(3)) ** 4 - 1.0) * 100.0

    return out


def add_missing_driver_proxies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "policy_rate" not in out.columns:
        yoy = out["annual_inflation_from_cpi"].bfill().fillna(0)
        mom = out["monthly_inflation_from_cpi"].fillna(0)
        out["policy_rate"] = np.maximum(5, 12 + 0.06 * yoy + 0.03 * mom)

    if "credit_private_sector_growth" not in out.columns:
        m3g = out["m3_growth"] if "m3_growth" in out.columns else pd.Series(0, index=out.index)
        reserve_g = out["reserve_money_growth"] if "reserve_money_growth" in out.columns else pd.Series(0, index=out.index)
        out["credit_private_sector_growth"] = 0.55 * m3g.fillna(0) + 0.15 * reserve_g.fillna(0)

    if "wage_growth_proxy" not in out.columns:
        out["wage_growth_proxy"] = (
            0.60 * out["monthly_inflation_from_cpi"].fillna(0)
            + 0.20 * out["core_monthly_inflation"].fillna(0)
        )

    if "output_gap_proxy" not in out.columns:
        m3g = out["m3_growth"] if "m3_growth" in out.columns else pd.Series(0, index=out.index)
        out["output_gap_proxy"] = m3g.fillna(0).rolling(3).mean().fillna(0) / 10.0

    if "fiscal_impulse_proxy" not in out.columns:
        reserve_g = out["reserve_money_growth"] if "reserve_money_growth" in out.columns else pd.Series(0, index=out.index)
        m3g = out["m3_growth"] if "m3_growth" in out.columns else pd.Series(0, index=out.index)
        out["fiscal_impulse_proxy"] = 0.35 * reserve_g.fillna(0) + 0.30 * m3g.fillna(0)

    if "exchange_rate_volatility_3m" not in out.columns:
        if "exrate_change" in out.columns:
            out["exchange_rate_volatility_3m"] = out["exrate_change"].rolling(3).std()
        else:
            out["exchange_rate_volatility_3m"] = 0.0

    if "inflation_expectations_proxy" not in out.columns:
        exr = out["exrate_change"] if "exrate_change" in out.columns else pd.Series(0, index=out.index)
        fuel = out["fuel_change"] if "fuel_change" in out.columns else pd.Series(0, index=out.index)
        m3g = out["m3_growth"] if "m3_growth" in out.columns else pd.Series(0, index=out.index)
        out["inflation_expectations_proxy"] = (
            0.30 * exr.fillna(0)
            + 0.20 * fuel.fillna(0)
            + 0.15 * m3g.fillna(0)
            + 0.20 * out["monthly_inflation_from_cpi"].shift(1).fillna(0)
            + 0.15 * out["core_monthly_inflation"].fillna(0)
        )

    if "imported_inflation_proxy" not in out.columns:
        exr = out["exrate_change"] if "exrate_change" in out.columns else pd.Series(0, index=out.index)
        fuel = out["fuel_change"] if "fuel_change" in out.columns else pd.Series(0, index=out.index)
        ppi = out["ppi_change"] if "ppi_change" in out.columns else pd.Series(0, index=out.index)
        cempi = out["cempi_change"] if "cempi_change" in out.columns else pd.Series(0, index=out.index)
        out["imported_inflation_proxy"] = (
            0.45 * exr.fillna(0)
            + 0.25 * fuel.fillna(0)
            + 0.15 * ppi.fillna(0)
            + 0.15 * cempi.fillna(0)
        )

    if "inflation_diffusion_proxy" not in out.columns:
        pieces = []
        for col in ["ppi_change", "core_monthly_inflation", "exrate_change", "fuel_change"]:
            if col in out.columns:
                s = out[col].fillna(0)
                rng = s.max() - s.min()
                normed = (s - s.min()) / rng if rng != 0 else s * 0
                pieces.append(normed)
        out["inflation_diffusion_proxy"] = 0.0 if len(pieces) == 0 else pd.concat(pieces, axis=1).mean(axis=1) * 100

    return out


def add_required_lags(df: pd.DataFrame, cols: List[str], max_lag: int = 3) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            for lag in range(1, max_lag + 1):
                lag_col = f"{col}_lag{lag}"
                if lag_col not in out.columns:
                    out[lag_col] = out[col].shift(lag)
    return out


def classify_policy_regime(yoy: float) -> str:
    if pd.isna(yoy):
        return "Below Lower Band"
    if yoy < SADC_LOWER:
        return "Below Lower Band"
    if yoy <= SADC_UPPER:
        return "Within SADC Band"
    if yoy <= 15:
        return "Moderately Above Band"
    return "Far Above Band"


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = derive_cpi_measures(out)
    out = ensure_core_series(out)
    out = add_missing_driver_proxies(out)

    lag_targets = list(TARGET_OPTIONS.values()) + BASE_DRIVER_COLS + FACTOR_NAMES
    out = add_required_lags(out, lag_targets, max_lag=3)

    out["policy_regime"] = out["annual_inflation_from_cpi"].apply(classify_policy_regime)
    out["policy_regime_code"] = out["policy_regime"].map(REGIME_MAP)
    return out


# =========================================================
# FACTOR ENGINE
# =========================================================
def winsorize_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    if s.dropna().empty:
        return s
    return s.clip(lower=s.quantile(lower_q), upper=s.quantile(upper_q))


def winsorize_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = winsorize_series(out[c])
    return out


def varimax(Phi: np.ndarray, gamma: float = 1.0, q: int = 20, tol: float = 1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = np.linalg.svd(
            np.dot(
                Phi.T,
                Lambda**3 - (gamma / p) * np.dot(Lambda, np.diag(np.diag(Lambda.T @ Lambda))),
            )
        )
        R = u @ vh
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return Phi @ R, R


def extract_single_block_factor(
    df: pd.DataFrame,
    cols: List[str],
    factor_name: str,
    smooth_window: int = 3,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, Dict]:
    usable_cols = [c for c in cols if c in df.columns]
    if len(usable_cols) < 2:
        empty = pd.Series(np.nan, index=df.index, name=factor_name)
        return empty, pd.DataFrame(), pd.DataFrame(), {}

    work = df[usable_cols].copy()
    for c in usable_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    valid_idx = work.dropna(how="all").index
    if len(valid_idx) < 12:
        empty = pd.Series(np.nan, index=df.index, name=factor_name)
        return empty, pd.DataFrame(), pd.DataFrame(), {}

    work_valid = work.loc[valid_idx].copy()
    work_valid = winsorize_df(work_valid, usable_cols)
    work_valid = sanitize_numeric_frame(work_valid)

    scaler = StandardScaler()
    X = scaler.fit_transform(work_valid)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_components = min(len(usable_cols), 3)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    loadings = pca.components_.T

    rotated_loadings, rotation_matrix = varimax(loadings)
    rotated_scores = scores @ rotation_matrix

    factor = pd.Series(np.nan, index=df.index, name=factor_name)
    factor.loc[valid_idx] = rotated_scores[:, 0]
    factor = factor.rolling(smooth_window, min_periods=1).mean()

    loadings_df = pd.DataFrame(
        rotated_loadings,
        index=usable_cols,
        columns=[f"PC{i+1}" for i in range(rotated_loadings.shape[1])],
    ).reset_index().rename(columns={"index": "variable"})
    loadings_df.insert(0, "factor_block", factor_name)

    explained_df = pd.DataFrame(
        {
            "factor_block": factor_name,
            "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )

    meta = {
        "usable_cols": usable_cols,
        "scaler": scaler,
        "pca": pca,
        "rotation_matrix": rotation_matrix,
        "smooth_window": smooth_window,
        "history": factor.dropna().tolist(),
    }

    return factor, loadings_df, explained_df, meta


def build_central_bank_factor_block(df: pd.DataFrame):
    out = df.copy()
    loadings_list = []
    explained_list = []
    factor_meta = {}

    for factor_name, cols in FACTOR_BLOCKS.items():
        try:
            factor_series, loadings_df, explained_df, meta = extract_single_block_factor(
                out, cols, factor_name, smooth_window=3
            )
            out[factor_name] = factor_series

            if not loadings_df.empty:
                loadings_list.append(loadings_df)
            if not explained_df.empty:
                explained_list.append(explained_df)

            factor_meta[factor_name] = meta
        except Exception:
            out[factor_name] = np.nan
            factor_meta[factor_name] = {}

    loadings_all = pd.concat(loadings_list, ignore_index=True) if loadings_list else pd.DataFrame()
    explained_all = pd.concat(explained_list, ignore_index=True) if explained_list else pd.DataFrame()

    return out, loadings_all, explained_all, factor_meta


def compute_factor_from_single_row(row: pd.Series, factor_name: str, factor_meta: Dict) -> float:
    meta = factor_meta.get(factor_name, {})
    usable_cols = meta.get("usable_cols", [])
    scaler = meta.get("scaler", None)
    pca = meta.get("pca", None)
    rotation_matrix = meta.get("rotation_matrix", None)
    smooth_window = meta.get("smooth_window", 3)
    history = meta.get("history", [])

    if not usable_cols or scaler is None or pca is None or rotation_matrix is None:
        return 0.0

    vals = []
    for c in usable_cols:
        vals.append(pd.to_numeric(row.get(c, 0.0), errors="coerce"))

    x = np.array(vals, dtype=float).reshape(1, -1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    z = scaler.transform(x)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    raw_scores = pca.transform(z)
    rotated_scores = raw_scores @ rotation_matrix
    current_factor = float(rotated_scores[0, 0])

    if smooth_window <= 1:
        return current_factor

    hist = history[-(smooth_window - 1):] if len(history) >= (smooth_window - 1) else history
    return float(np.mean(hist + [current_factor]))


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def build_feature_set(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()

    if target_col not in out.columns:
        raise ValueError(f"Target column '{target_col}' is missing.")

    predictor_cols = [c for c in FACTOR_NAMES if c in out.columns]
    direct_controls = [
        "policy_rate",
        "exrate_change",
        "fuel_change",
        "fiscal_impulse_proxy",
        "inflation_expectations_proxy",
        "imported_inflation_proxy",
        "exchange_rate_volatility_3m",
    ]
    predictor_cols += [c for c in direct_controls if c in out.columns]

    for lag in range(1, 4):
        lag_col = f"{target_col}_lag{lag}"
        if lag_col in out.columns:
            predictor_cols.append(lag_col)

    for factor_col in FACTOR_NAMES:
        for lag in range(1, 4):
            lag_col = f"{factor_col}_lag{lag}"
            if lag_col in out.columns:
                predictor_cols.append(lag_col)

    predictor_cols = list(dict.fromkeys(predictor_cols))

    needed = ["date", target_col] + predictor_cols
    model_df = out[needed].copy()

    numeric_cols = [c for c in model_df.columns if c != "date"]
    model_df[numeric_cols] = sanitize_numeric_frame(model_df[numeric_cols])

    model_df = model_df.dropna(subset=["date", target_col]).copy()
    if model_df.empty:
        raise ValueError(f"No usable rows remain for target '{target_col}' after preparation.")

    return model_df, predictor_cols


# =========================================================
# MODEL HELPERS
# =========================================================
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_regression(y_true, y_pred, model_name: str) -> Dict[str, float]:
    return {
        "Model": model_name,
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def get_model_registry() -> Dict[str, object]:
    registry = {
        "Elastic Net": ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            cv=5,
            random_state=123,
            max_iter=20000,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            random_state=123,
            min_samples_leaf=2,
        ),
        "SVR": SVR(kernel="rbf", C=5.0, epsilon=0.05),
    }

    if XGBOOST_AVAILABLE:
        registry["XGBoost"] = XGBRegressor(
            n_estimators=250,
            objective="reg:squarederror",
            random_state=123,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
        )
    return registry


def compute_linear_contributions(
    model_name: str,
    model,
    x_row: pd.DataFrame,
    predictor_cols: List[str],
) -> pd.DataFrame:
    if model_name != "Elastic Net":
        return pd.DataFrame(
            {
                "Component": ["Model Type"],
                "Contribution": [np.nan],
                "Note": ["Contribution breakdown shown only for Elastic Net benchmark"],
            }
        )

    coef = pd.Series(model.coef_, index=predictor_cols)
    contributions = coef * x_row.iloc[0][predictor_cols]
    intercept = float(model.intercept_)

    table = pd.DataFrame(
        {
            "Component": list(contributions.index) + ["Intercept"],
            "Contribution": list(contributions.values) + [intercept],
        }
    ).sort_values("Contribution", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    table["Note"] = ""
    return table


def fit_regression_models(
    model_df: pd.DataFrame,
    predictor_cols: List[str],
    target_col: str,
    train_ratio: float,
) -> ForecastBundle:
    if len(model_df) < 24:
        raise ValueError(f"Need at least 24 usable observations for forecasting '{target_col}'.")

    train_size = max(18, int(len(model_df) * train_ratio))
    if train_size >= len(model_df):
        train_size = len(model_df) - 1

    train_df = model_df.iloc[:train_size].copy()
    test_df = model_df.iloc[train_size:].copy()

    x_train = sanitize_numeric_frame(train_df[predictor_cols])
    x_test = sanitize_numeric_frame(test_df[predictor_cols])
    y_train = pd.to_numeric(train_df[target_col], errors="coerce").fillna(0.0).values
    y_test = pd.to_numeric(test_df[target_col], errors="coerce").fillna(0.0).values

    preds = {}
    fitted = {}
    resid_stds = {}

    for name, model in get_model_registry().items():
        if name in ["Elastic Net", "XGBoost"]:
            model.fit(x_train.values, y_train)
            pred_test = model.predict(x_test.values)
            pred_train = model.predict(x_train.values)
        else:
            model.fit(x_train, y_train)
            pred_test = model.predict(x_test)
            pred_train = model.predict(x_train)

        preds[name] = pred_test
        fitted[name] = model

        resid = y_train - pred_train
        resid_std = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.10
        resid_stds[name] = max(resid_std, 0.10)

    metrics = pd.DataFrame(
        [evaluate_regression(y_test, preds[name], name) for name in preds.keys()]
    ).sort_values("RMSE").reset_index(drop=True)

    best_model_name = metrics.iloc[0]["Model"]

    backtest = pd.DataFrame({"date": test_df["date"].values, "Actual": y_test})
    for name, pred in preds.items():
        backtest[name] = pred
    backtest["Best Prediction"] = backtest[best_model_name]

    contribution_table = compute_linear_contributions(
        "Elastic Net",
        fitted["Elastic Net"],
        x_test.tail(1),
        predictor_cols,
    )

    return ForecastBundle(
        model_df=model_df,
        metrics=metrics,
        backtest=backtest,
        best_model_name=best_model_name,
        predictor_cols=predictor_cols,
        fitted_models=fitted,
        residual_std=resid_stds[best_model_name],
        contribution_table=contribution_table,
    )


def fit_regime_classifier(df: pd.DataFrame, predictor_cols: List[str], train_ratio: float) -> RegimeBundle:
    cols = ["policy_regime_code", "date", "annual_inflation_from_cpi"] + predictor_cols
    work = df[cols].copy()

    numeric_cols = [c for c in work.columns if c not in ["date"]]
    work[numeric_cols] = sanitize_numeric_frame(work[numeric_cols])
    work = work.dropna(subset=["date"]).copy()

    if len(work) < 24:
        raise ValueError("Need at least 24 usable observations for regime classification.")

    train_size = max(18, int(len(work) * train_ratio))
    if train_size >= len(work):
        train_size = len(work) - 1

    train_df = work.iloc[:train_size].copy()
    test_df = work.iloc[train_size:].copy()

    x_train = train_df[predictor_cols]
    x_test = test_df[predictor_cols]
    y_train = train_df["policy_regime_code"].astype(int)
    y_test = test_df["policy_regime_code"].astype(int)

    if len(np.unique(y_train)) < 2:
        raise ValueError("Regime classifier needs at least two classes in the training sample.")

    cls = RandomForestClassifier(
        n_estimators=300,
        random_state=123,
        min_samples_leaf=1,
    )
    cls.fit(x_train, y_train)

    pred = cls.predict(x_test)
    probs = cls.predict_proba(x_test)

    class_labels = list(cls.classes_)
    prob_names = [f"Prob_{REVERSE_REGIME_MAP[c]}" for c in class_labels]
    prob_df = pd.DataFrame(probs, columns=prob_names)

    class_results = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": test_df["date"].values,
                    "actual_regime": pd.Series(y_test).map(REVERSE_REGIME_MAP).values,
                    "predicted_regime": pd.Series(pred).map(REVERSE_REGIME_MAP).values,
                    "annual_inflation": test_df["annual_inflation_from_cpi"].values,
                }
            ).reset_index(drop=True),
            prob_df.reset_index(drop=True),
        ],
        axis=1,
    )

    labels = sorted(np.unique(np.concatenate([y_test.values, pred])))
    cm = confusion_matrix(y_test, pred, labels=labels)

    confusion = pd.DataFrame(
        cm,
        index=[f"Actual {REVERSE_REGIME_MAP[i]}" for i in labels],
        columns=[f"Pred {REVERSE_REGIME_MAP[i]}" for i in labels],
    )

    report = classification_report(
        y_test,
        pred,
        labels=labels,
        target_names=[REVERSE_REGIME_MAP[i] for i in labels],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "Class"})

    return RegimeBundle(
        class_results=class_results,
        confusion=confusion,
        report=report_df,
        accuracy=float(accuracy_score(y_test, pred)),
    )


# =========================================================
# NOWCASTING AND FUTURE PATH
# =========================================================
def build_shocked_last_row(df_factor: pd.DataFrame, scenario_shocks: Dict[str, float]) -> pd.Series:
    row = df_factor.iloc[-1].copy()
    for col, shock in scenario_shocks.items():
        if col in row.index and pd.notna(row[col]):
            row[col] = float(row[col]) + float(shock)
        elif col in BASE_DRIVER_COLS:
            row[col] = float(shock)
    return row


def recompute_row_factors(row: pd.Series, factor_meta: Dict) -> Dict[str, float]:
    out = {}
    for factor_name in FACTOR_NAMES:
        out[factor_name] = compute_factor_from_single_row(row, factor_name, factor_meta)
    return out


def make_nowcast_row(
    df_factor: pd.DataFrame,
    target_col: str,
    predictor_cols: List[str],
    scenario_shocks: Dict[str, float],
    factor_meta: Dict,
) -> pd.DataFrame:
    working = df_factor.copy()
    shocked = build_shocked_last_row(working, scenario_shocks)
    factor_vals = recompute_row_factors(shocked, factor_meta)

    row = {"date": pd.to_datetime(working["date"].iloc[-1])}

    for col in BASE_DRIVER_COLS:
        row[col] = float(pd.to_numeric(shocked.get(col, 0.0), errors="coerce") if pd.notna(shocked.get(col, np.nan)) else 0.0)

    for fcol in FACTOR_NAMES:
        row[fcol] = float(factor_vals.get(fcol, 0.0))

    target_history = working[target_col].dropna()
    target_last = float(target_history.iloc[-1]) if not target_history.empty else 0.0

    for lag in range(1, 4):
        lag_col = f"{target_col}_lag{lag}"
        if lag == 1:
            row[lag_col] = target_last
        else:
            row[lag_col] = row.get(f"{target_col}_lag{lag-1}", target_last)

    for fcol in FACTOR_NAMES:
        factor_history = working[fcol].dropna() if fcol in working.columns else pd.Series(dtype=float)
        factor_last = float(factor_history.iloc[-1]) if not factor_history.empty else 0.0
        for lag in range(1, 4):
            lag_col = f"{fcol}_lag{lag}"
            if lag == 1:
                row[lag_col] = factor_last
            else:
                row[lag_col] = row.get(f"{fcol}_lag{lag-1}", factor_last)

    x_now = pd.DataFrame([row]).reindex(columns=predictor_cols, fill_value=0.0)
    x_now = sanitize_numeric_frame(x_now)
    return x_now


def generate_nowcast(
    df_factor: pd.DataFrame,
    target_col: str,
    forecast_bundle: ForecastBundle,
    scenario_shocks: Dict[str, float],
    factor_meta: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x_now = make_nowcast_row(
        df_factor,
        target_col,
        forecast_bundle.predictor_cols,
        scenario_shocks,
        factor_meta,
    )

    rows = []
    for name, model in forecast_bundle.fitted_models.items():
        if name in ["Elastic Net", "XGBoost"]:
            pred = float(model.predict(x_now.values)[0])
        else:
            pred = float(model.predict(x_now)[0])
        rows.append({"Model": name, "Nowcast": pred})

    out = pd.DataFrame(rows)
    out["Best Model"] = np.where(out["Model"] == forecast_bundle.best_model_name, "Yes", "")
    out = out.sort_values("Nowcast").reset_index(drop=True)

    contribution_table = compute_linear_contributions(
        "Elastic Net",
        forecast_bundle.fitted_models["Elastic Net"],
        x_now,
        forecast_bundle.predictor_cols,
    )

    return out, contribution_table


def simple_future_path(
    df_factor: pd.DataFrame,
    selected_target_col: str,
    forecast_bundle: ForecastBundle,
    horizon: int,
    scenario_shocks: Dict[str, float],
    factor_meta: Dict,
) -> pd.DataFrame:
    model = forecast_bundle.fitted_models[forecast_bundle.best_model_name]
    working = forecast_bundle.model_df.copy()
    predictor_cols = forecast_bundle.predictor_cols

    future_rows = []
    current_last_date = pd.to_datetime(working["date"].iloc[-1])

    for step in range(1, horizon + 1):
        base_row = build_shocked_last_row(working, scenario_shocks)
        factor_vals = recompute_row_factors(base_row, factor_meta)

        row = {"date": current_last_date + pd.DateOffset(months=step)}

        for col in BASE_DRIVER_COLS:
            row[col] = float(pd.to_numeric(base_row.get(col, 0.0), errors="coerce") if pd.notna(base_row.get(col, np.nan)) else 0.0)

        for fcol in FACTOR_NAMES:
            row[fcol] = float(factor_vals.get(fcol, 0.0))

        target_hist = working[selected_target_col].dropna()
        target_last = float(target_hist.iloc[-1]) if not target_hist.empty else 0.0
        for lag in range(1, 4):
            lag_col = f"{selected_target_col}_lag{lag}"
            if lag == 1:
                row[lag_col] = target_last
            else:
                row[lag_col] = row.get(f"{selected_target_col}_lag{lag-1}", target_last)

        for fcol in FACTOR_NAMES:
            factor_hist = working[fcol].dropna() if fcol in working.columns else pd.Series(dtype=float)
            factor_last = float(factor_hist.iloc[-1]) if not factor_hist.empty else 0.0
            for lag in range(1, 4):
                lag_col = f"{fcol}_lag{lag}"
                if lag == 1:
                    row[lag_col] = factor_last
                else:
                    row[lag_col] = row.get(f"{fcol}_lag{lag-1}", factor_last)

        x_row = pd.DataFrame([row]).reindex(columns=predictor_cols, fill_value=0.0)
        x_row = sanitize_numeric_frame(x_row)

        if forecast_bundle.best_model_name in ["Elastic Net", "XGBoost"]:
            pred = float(model.predict(x_row.values)[0])
        else:
            pred = float(model.predict(x_row)[0])

        row[selected_target_col] = pred

        lower_68 = pred - forecast_bundle.residual_std
        upper_68 = pred + forecast_bundle.residual_std
        lower_95 = pred - 1.96 * forecast_bundle.residual_std
        upper_95 = pred + 1.96 * forecast_bundle.residual_std

        future_rows.append(
            {
                "date": row["date"],
                "forecast": pred,
                "lower_68": lower_68,
                "upper_68": upper_68,
                "lower_95": lower_95,
                "upper_95": upper_95,
                "step": step,
                **{f: row.get(f, np.nan) for f in FACTOR_NAMES},
            }
        )

        working = pd.concat([working, pd.DataFrame([row])], ignore_index=True)

    future_df = pd.DataFrame(future_rows)
    if future_df.empty:
        return pd.DataFrame(columns=["date", "forecast", "lower_68", "upper_68", "lower_95", "upper_95", "step"])
    return future_df


# =========================================================
# CHARTS
# =========================================================
def band_chart(df: pd.DataFrame, series_col: str, title: str):
    fig = go.Figure()
    fig.add_hrect(y0=SADC_LOWER, y1=SADC_UPPER, fillcolor="lightgreen", opacity=0.20, line_width=0)
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df[series_col],
            mode="lines",
            name=series_col,
            line=dict(width=3, color="#174EA6"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=380,
        title={"text": title, "x": 0.02, "xanchor": "left"},
        font=dict(color="#132238"),
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Date",
        yaxis_title="Inflation (%)",
        hovermode="x unified",
    )
    return fig


def backtest_chart(backtest_df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=backtest_df["date"],
            y=backtest_df["Actual"],
            mode="lines+markers",
            name="Actual",
            line=dict(width=3, color="#0B1F3A"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=backtest_df["date"],
            y=backtest_df["Best Prediction"],
            mode="lines+markers",
            name="Best Prediction",
            line=dict(width=3, color="#D4A017"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=400,
        title={"text": title, "x": 0.02, "xanchor": "left"},
        font=dict(color="#132238"),
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Date",
        yaxis_title="Inflation (%)",
        hovermode="x unified",
    )
    return fig


def confusion_chart(cm: pd.DataFrame):
    fig = px.imshow(cm, text_auto=True, aspect="auto", template="plotly_white")
    fig.update_layout(
        height=380,
        title={"text": "Policy Regime Confusion Matrix", "x": 0.02, "xanchor": "left"},
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def fan_chart(history_df: pd.DataFrame, future_df: pd.DataFrame, target_col: str, title: str):
    fig = go.Figure()
    fig.add_hrect(y0=SADC_LOWER, y1=SADC_UPPER, fillcolor="lightgreen", opacity=0.15, line_width=0)
    fig.add_trace(
        go.Scatter(
            x=history_df["date"],
            y=history_df[target_col],
            mode="lines",
            name="Historical",
            line=dict(width=3, color="#0B1F3A"),
        )
    )

    if not future_df.empty and {"date", "upper_95", "lower_95", "upper_68", "lower_68", "forecast"}.issubset(future_df.columns):
        fig.add_trace(go.Scatter(x=future_df["date"], y=future_df["upper_95"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=future_df["date"], y=future_df["lower_95"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(23,78,166,0.12)", name="95% band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=future_df["date"], y=future_df["upper_68"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=future_df["date"], y=future_df["lower_68"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(212,160,23,0.25)", name="68% band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=future_df["date"], y=future_df["forecast"], mode="lines+markers", name="Forecast", line=dict(width=3, color="#D4A017")))

    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.02, "xanchor": "left"},
        height=460,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Date",
        yaxis_title="Inflation (%)",
        hovermode="x unified",
    )
    return fig


def factor_path_chart(df: pd.DataFrame):
    fig = go.Figure()
    palette = ["#174EA6", "#D4A017", "#1F6FB2", "#0B1F3A"]
    for i, f in enumerate(FACTOR_NAMES):
        if f in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df[f],
                    mode="lines",
                    name=f.replace("_", " ").title(),
                    line=dict(width=2.5, color=palette[i % len(palette)]),
                )
            )
    fig.update_layout(
        template="plotly_white",
        height=420,
        title={"text": "Central Bank Factor Paths", "x": 0.02, "xanchor": "left"},
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Date",
        yaxis_title="Factor Score",
        hovermode="x unified",
    )
    return fig


def loadings_bar_chart(loadings_all: pd.DataFrame, selected_factor: str):
    block = loadings_all[loadings_all["factor_block"] == selected_factor].copy()
    if block.empty:
        return go.Figure()
    block = block.sort_values("PC1", key=lambda s: s.abs(), ascending=False)

    fig = px.bar(
        block,
        x="variable",
        y="PC1",
        template="plotly_white",
        title=f"{selected_factor.replace('_', ' ').title()} – Rotated PC1 Loadings",
    )
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Variable",
        yaxis_title="Loading",
    )
    return fig


def contribution_bar_chart(contribution_table: pd.DataFrame, title: str):
    work = contribution_table.copy()
    work = work[np.isfinite(work["Contribution"])].copy()
    if work.empty:
        return go.Figure()

    work = work.sort_values("Contribution", key=lambda s: s.abs(), ascending=False).head(12)
    fig = px.bar(
        work,
        x="Contribution",
        y="Component",
        orientation="h",
        template="plotly_white",
        title=title,
    )
    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis={"categoryorder": "total ascending"},
    )
    return fig


# =========================================================
# EXPORT HELPERS
# =========================================================
def scenario_summary_text(
    scenario_name: str,
    shocks: Dict[str, float],
    nowcast_df: pd.DataFrame,
    target_label: str,
) -> str:
    best_row = nowcast_df.loc[nowcast_df["Best Model"] == "Yes"]
    best_nowcast = float(best_row["Nowcast"].iloc[0]) if not best_row.empty else float(nowcast_df["Nowcast"].mean())

    non_zero = {k: v for k, v in shocks.items() if abs(v) > 1e-12}
    if non_zero:
        drivers = ", ".join([f"{k} {v:+.1f}" for k, v in non_zero.items()])
    else:
        drivers = "no active shock adjustments"

    band_status = (
        "below the lower SADC band"
        if best_nowcast < SADC_LOWER
        else "within the SADC target band"
        if best_nowcast <= SADC_UPPER
        else "above the SADC target band"
    )

    return (
        f"Scenario: {scenario_name}. "
        f"The best-model {target_label.lower()} nowcast is {best_nowcast:.2f} percent, "
        f"placing projected inflation {band_status}. "
        f"Scenario assumptions applied: {drivers}."
    )


def dataframe_to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    engine = "xlsxwriter"
    try:
        import xlsxwriter  # noqa: F401
    except Exception:
        engine = "openpyxl"

    with pd.ExcelWriter(output, engine=engine) as writer:
        for name, df in sheets.items():
            safe_name = name[:31]
            df.to_excel(writer, index=False, sheet_name=safe_name)

            if engine == "xlsxwriter":
                worksheet = writer.sheets[safe_name]
                for idx, col in enumerate(df.columns):
                    col_width = min(max(len(str(col)) + 3, 14), 30)
                    worksheet.set_column(idx, idx, col_width)

    output.seek(0)
    return output.getvalue()


# =========================================================
# PIPELINE
# =========================================================
@st.cache_data(show_spinner=False)
def run_pipeline_from_source(source) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    raw = safe_read_csv(source)
    prepared = prepare_dataset(raw)
    factored, loadings_all, explained_all, factor_meta = build_central_bank_factor_block(prepared)
    factored = add_required_lags(factored, FACTOR_NAMES, max_lag=3)

    numeric_cols = [c for c in factored.columns if c != "date"]
    factored[numeric_cols] = sanitize_numeric_frame(factored[numeric_cols])

    return factored, loadings_all, explained_all, factor_meta


def latest_regime_label(value: float) -> str:
    return classify_policy_regime(value)


# =========================================================
# APP HEADER
# =========================================================
st.markdown(
    """
    <div class="rbz-hero">
        <h1>Factor-Augmented Inflation Forecasting Framework</h1>
        <p>Central bank-grade inflation nowcasting, multi-model forecasting, regime classification, and scenario analysis.</p>
        <div class="rbz-subline">
            Factor blocks: Domestic Liquidity • External Cost • Inflation Momentum • Demand/Fiscal Conditions
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## System Controls")
    uploaded_file = st.file_uploader("Upload inflation dataset (CSV)", type=["csv"])

    st.markdown("## Forecast Settings")
    selected_target_label = st.selectbox("Target inflation measure", list(TARGET_OPTIONS.keys()), index=1)
    selected_target_col = TARGET_OPTIONS[selected_target_label]

    train_ratio = st.slider("Training sample ratio", min_value=0.60, max_value=0.90, value=0.80, step=0.05)
    forecast_horizon = st.slider("Forecast horizon (months)", min_value=3, max_value=18, value=12, step=1)

    st.markdown("## Scenario Design")
    scenario_name = st.selectbox("Scenario preset", list(SCENARIO_LIBRARY.keys()), index=0)
    preset = SCENARIO_LIBRARY[scenario_name].copy()

    custom_mode = st.checkbox("Enable custom scenario adjustments", value=False)
    if custom_mode:
        scenario_shocks = {}
        for col, default_val in preset.items():
            scenario_shocks[col] = st.number_input(
                f"{col}",
                value=float(default_val),
                step=0.5,
                format="%.2f",
            )
    else:
        scenario_shocks = preset

    st.markdown("## Data Source")
    use_default_data = st.checkbox("Fallback to default local CSV path", value=uploaded_file is None)


# =========================================================
# DATA LOAD
# =========================================================
try:
    if uploaded_file is not None:
        df_factor, loadings_all, explained_all, factor_meta = run_pipeline_from_source(uploaded_file)
        source_note = "Uploaded CSV"
    elif use_default_data:
        df_factor, loadings_all, explained_all, factor_meta = run_pipeline_from_source(DEFAULT_DATA_PATH)
        source_note = DEFAULT_DATA_PATH
    else:
        st.info("Upload a CSV file or enable the local default-path option in the sidebar.")
        st.stop()
except FileNotFoundError:
    st.error(f"Default file not found at: {DEFAULT_DATA_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()


# =========================================================
# MODEL FITTING
# =========================================================
future_df = pd.DataFrame()

try:
    model_df, predictor_cols = build_feature_set(df_factor, selected_target_col)

    forecast_bundle = fit_regression_models(
        model_df=model_df,
        predictor_cols=predictor_cols,
        target_col=selected_target_col,
        train_ratio=train_ratio,
    )

    regime_bundle = fit_regime_classifier(
        df=df_factor,
        predictor_cols=predictor_cols,
        train_ratio=train_ratio,
    )

    nowcast_df, nowcast_contrib = generate_nowcast(
        df_factor=df_factor,
        target_col=selected_target_col,
        forecast_bundle=forecast_bundle,
        scenario_shocks=scenario_shocks,
        factor_meta=factor_meta,
    )

    future_df = simple_future_path(
        df_factor=df_factor,
        selected_target_col=selected_target_col,
        forecast_bundle=forecast_bundle,
        horizon=forecast_horizon,
        scenario_shocks=scenario_shocks,
        factor_meta=factor_meta,
    )

except Exception as e:
    st.error(f"Pipeline execution failed: {e}")
    st.stop()


# =========================================================
# TOP METRICS
# =========================================================
latest_actual_target = float(df_factor[selected_target_col].dropna().iloc[-1])
latest_yoy = float(df_factor["annual_inflation_from_cpi"].dropna().iloc[-1])
best_nowcast = float(nowcast_df.loc[nowcast_df["Best Model"] == "Yes", "Nowcast"].iloc[0])
best_rmse = float(forecast_bundle.metrics.iloc[0]["RMSE"])
latest_regime = latest_regime_label(latest_yoy)

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Latest Actual", f"{latest_actual_target:,.2f}%", selected_target_label)
with c2:
    metric_card("Best Nowcast", f"{best_nowcast:,.2f}%", forecast_bundle.best_model_name)
with c3:
    metric_card("Backtest RMSE", f"{best_rmse:,.2f}", "Lower is better")
with c4:
    metric_card("Current Regime", latest_regime, f"Annual inflation: {latest_yoy:.2f}%")


# =========================================================
# POLICY SUMMARY
# =========================================================
summary_text = scenario_summary_text(
    scenario_name=scenario_name,
    shocks=scenario_shocks,
    nowcast_df=nowcast_df,
    target_label=selected_target_label,
)

st.markdown(
    f"""
    <div class="policy-card">
        <h3>Monetary Policy Brief</h3>
        <p><b>Data source:</b> {source_note}</p>
        <p><b>Selected target:</b> {selected_target_label}</p>
        <p><b>Best-performing forecasting model:</b> {forecast_bundle.best_model_name}</p>
        <p>{summary_text}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Executive Dashboard",
        "Factor Engine",
        "Forecasting & Nowcast",
        "Policy Regimes",
        "Data & Downloads",
        "Model Documentation",
        "Disclaimer and Ownership Notice",
    ]
)

# =========================================================
# TAB 1
# =========================================================
with tab1:
    left, right = st.columns([1.25, 1])

    with left:
        st.plotly_chart(
            band_chart(
                df_factor.dropna(subset=[selected_target_col]),
                selected_target_col,
                f"{selected_target_label} – Historical Path",
            ),
            width="stretch",
        )

    with right:
        st.plotly_chart(
            factor_path_chart(df_factor),
            width="stretch",
        )

    left, right = st.columns([1, 1])

    with left:
        st.plotly_chart(
            backtest_chart(
                forecast_bundle.backtest,
                f"Backtest Performance – {forecast_bundle.best_model_name}",
            ),
            width="stretch",
        )

    with right:
        if not future_df.empty and "date" in future_df.columns:
            st.plotly_chart(
                fan_chart(
                    forecast_bundle.model_df,
                    future_df,
                    selected_target_col,
                    f"Forecast Fan Chart – {selected_target_label}",
                ),
                width="stretch",
            )
        else:
            st.warning("Forecast path could not be generated.")

    st.markdown("### Forecast Summary Table")
    if not future_df.empty and "date" in future_df.columns:
        forecast_summary = future_df[
            ["date", "forecast", "lower_68", "upper_68", "lower_95", "upper_95"]
        ].copy()
        forecast_summary["date"] = pd.to_datetime(forecast_summary["date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(forecast_summary, width="stretch", hide_index=True)
    else:
        st.info("No forecast summary available.")


# =========================================================
# TAB 2
# =========================================================
with tab2:
    st.markdown("### Rotated Central Bank Factor Structure")
    st.markdown(
        """
The framework extracts four interpretable latent blocks:
domestic liquidity conditions, external cost pressures, inflation momentum, and demand/fiscal impulse.
Factors are estimated using standardized data, PCA, and varimax rotation.
"""
    )

    c1, c2 = st.columns([1.05, 1])

    with c1:
        st.dataframe(explained_all, width="stretch", hide_index=True)

    with c2:
        selected_factor = st.selectbox(
            "Select factor block",
            FACTOR_NAMES,
            format_func=lambda x: x.replace("_", " ").title(),
        )
        st.plotly_chart(loadings_bar_chart(loadings_all, selected_factor), width="stretch")

    st.markdown("### Factor Loadings Table")
    st.dataframe(loadings_all, width="stretch", hide_index=True)

    st.markdown("### Factor Time Series")
    factor_table = df_factor[["date"] + [f for f in FACTOR_NAMES if f in df_factor.columns]].copy()
    st.dataframe(factor_table, width="stretch", hide_index=True)


# =========================================================
# TAB 3
# =========================================================
with tab3:
    c1, c2 = st.columns([0.95, 1.05])

    with c1:
        st.markdown("### Model Performance")
        st.dataframe(forecast_bundle.metrics, width="stretch", hide_index=True)

        st.markdown("### Model Nowcast Table")
        st.dataframe(nowcast_df, width="stretch", hide_index=True)

        st.markdown("### Elastic Net Driver Contribution")
        st.dataframe(nowcast_contrib, width="stretch", hide_index=True)

    with c2:
        st.plotly_chart(
            contribution_bar_chart(nowcast_contrib, "Elastic Net Driver Contributions to Current Nowcast"),
            width="stretch",
        )

    st.markdown("### Scenario Settings Applied")
    scenario_df = pd.DataFrame(
        {
            "driver": list(scenario_shocks.keys()),
            "shock": list(scenario_shocks.values()),
        }
    )
    st.dataframe(scenario_df, width="stretch", hide_index=True)


# =========================================================
# TAB 4
# =========================================================
with tab4:
    c1, c2 = st.columns([0.95, 1.05])

    with c1:
        metric_card("Regime Accuracy", f"{regime_bundle.accuracy:.2%}", "Random Forest classifier")
        st.markdown("### Classification Report")
        st.dataframe(regime_bundle.report, width="stretch", hide_index=True)

    with c2:
        st.plotly_chart(confusion_chart(regime_bundle.confusion), width="stretch")

    st.markdown("### Regime Prediction Results")
    class_results = regime_bundle.class_results.copy()
    class_results["date"] = pd.to_datetime(class_results["date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(class_results, width="stretch", hide_index=True)


# =========================================================
# TAB 5
# =========================================================
with tab5:
    st.markdown("### Prepared Modelling Dataset")
    preview_cols = ["date", "cpi", selected_target_col] + [c for c in FACTOR_NAMES if c in df_factor.columns]
    preview_cols = [c for c in preview_cols if c in df_factor.columns]
    st.dataframe(df_factor[preview_cols], width="stretch", hide_index=True)

    sheets = {
        "prepared_data": df_factor,
        "forecast_metrics": forecast_bundle.metrics,
        "backtest": forecast_bundle.backtest,
        "nowcast": nowcast_df,
        "forecast_path": future_df,
        "regime_results": regime_bundle.class_results,
        "regime_confusion": regime_bundle.confusion.reset_index(),
        "factor_loadings": loadings_all,
        "factor_explained": explained_all,
        "contributions": nowcast_contrib,
    }

    excel_bytes = dataframe_to_excel_bytes(sheets)

    st.download_button(
        label="Download Full Results Workbook",
        data=excel_bytes,
        file_name="rbz_factor_augmented_inflation_framework_outputs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    csv_bytes = df_factor.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Prepared Dataset (CSV)",
        data=csv_bytes,
        file_name="rbz_prepared_factor_dataset.csv",
        mime="text/csv",
    )
with tab6:
    st.title("Model Documentation")

    st.markdown(r"""
This application is a **factor-augmented inflation forecasting and nowcasting framework**
designed for policy analysis, scenario testing, and inflation regime assessment.

It combines:

- macro-financial driver preprocessing,
- proxy construction for partially observed macroeconomic conditions,
- latent factor extraction using Principal Component Analysis (PCA),
- multi-model supervised forecasting,
- inflation regime classification,
- and scenario-conditioned forward projections.

At a high level, the framework maps observed macroeconomic data into a structured prediction system:
""")

    st.latex(r"""
\text{Raw Data}
\;\longrightarrow\;
\text{Preprocessing and Proxy Construction}
\;\longrightarrow\;
\text{Factor Extraction}
\;\longrightarrow\;
\text{Forecast Models}
\;\longrightarrow\;
\text{Nowcast, Forecast Path, Regime, and Scenarios}
""")

    with st.expander("1. What the model does", expanded=True):
        st.markdown(r"""
The system performs five core functions:

### 1. Inflation measurement
It constructs inflation indicators from CPI data, including:
- monthly inflation,
- annual inflation,
- quarterly inflation,
- annualised quarterly inflation,
- and core inflation indicators where available or approximated.

### 2. Macro driver reconstruction
Where direct macroeconomic series are unavailable, the system constructs policy-relevant proxies for:
- monetary conditions,
- fiscal impulse,
- imported inflation,
- inflation expectations,
- wage pressure,
- diffusion of price pressures,
- and output conditions.

### 3. Latent factor extraction
The system groups macro variables into coherent economic blocks and reduces them into latent factor indices using PCA.
This creates compact summary measures of broader inflationary conditions.

### 4. Forecast generation
The model then estimates several alternative forecasting models using the same prepared feature matrix and compares their out-of-sample performance.

### 5. Policy interpretation
The system classifies inflation into policy-relevant regimes, produces scenario-conditioned forecasts, and generates uncertainty bands for operational interpretation.

In compact notation, the application seeks to approximate the mapping:
""")

        st.latex(r"""
\hat{\pi}_{t+h} = f\left(\pi_t,\pi_{t-1},\ldots,F_t,F_{t-1},\ldots,X_t\right)
""")

        st.markdown(r"""
where:

- \( \hat{\pi}_{t+h} \) is the forecast inflation rate at horizon \( h \),
- \( \pi_t \) denotes current or lagged inflation,
- \( F_t \) denotes latent factor scores,
- \( X_t \) denotes other direct macro-financial controls.
""")

    with st.expander("2. Mathematical framework", expanded=False):
        st.markdown(r"""
### 2.1 Inflation measurement

The model begins with the CPI series. Let \( CPI_t \) denote the Consumer Price Index at time \( t \).

#### Monthly inflation
""")

        st.latex(r"""
\pi^{m}_{t} = \left(\frac{CPI_t}{CPI_{t-1}} - 1\right)\times 100
""")

        st.markdown(r"""
This measures month-on-month inflation.

#### Annual inflation
""")

        st.latex(r"""
\pi^{y}_{t} = \left(\frac{CPI_t}{CPI_{t-12}} - 1\right)\times 100
""")

        st.markdown(r"""
This measures year-on-year inflation relative to the same month in the previous year.

#### Quarterly inflation
""")

        st.latex(r"""
\pi^{q}_{t} = \left(\frac{CPI_t}{CPI_{t-3}} - 1\right)\times 100
""")

        st.markdown(r"""
This measures quarter-on-quarter inflation using a three-month lag.

#### Annualised quarterly inflation
""")

        st.latex(r"""
\pi^{q,a}_{t} = \left[\left(\frac{CPI_t}{CPI_{t-3}}\right)^4 - 1\right]\times 100
""")

        st.markdown(r"""
This transforms quarterly inflation into an annualised rate.

### 2.2 Core inflation approximation

Where a direct core CPI series is unavailable, the model constructs an approximate core price index as a weighted combination of subcomponents:
""")

        st.latex(r"""
CoreCPI_t
=
w_1 NonFoodCPI_t
+
w_2 NonTradablesCPI_t
+
w_3 TradablesCPI_t
+
w_4 AdministeredPriceCPI_t
""")

        st.markdown(r"""
subject to:
""")

        st.latex(r"""
w_1 + w_2 + w_3 + w_4 = 1
""")

        st.markdown(r"""
Core inflation measures are then computed analogously:
""")

        st.latex(r"""
\pi^{core,m}_{t} = \left(\frac{CoreCPI_t}{CoreCPI_{t-1}} - 1\right)\times 100
""")

        st.latex(r"""
\pi^{core,y}_{t} = \left(\frac{CoreCPI_t}{CoreCPI_{t-12}} - 1\right)\times 100
""")

        st.markdown(r"""
### 2.3 Standardization of factor inputs

Suppose a factor block contains \( N \) variables observed over \( T \) periods.
Let \( x_{it} \) denote variable \( i \) at time \( t \).

Each variable is standardized before PCA:
""")

        st.latex(r"""
z_{it} = \frac{x_{it} - \mu_i}{\sigma_i}
""")

        st.markdown(r"""
where:

- \( \mu_i \) is the sample mean of variable \( i \),
- \( \sigma_i \) is the sample standard deviation of variable \( i \).

This ensures that variables measured in different units contribute on a comparable scale.

### 2.4 Factor representation

Let \( Z \) denote the standardized data matrix of dimension \( T \times N \).
The factor structure is written as:
""")

        st.latex(r"""
Z = F\Lambda' + U
""")

        st.markdown(r"""
where:

- \( Z \) is the standardized data matrix,
- \( F \) is the \( T \times K \) matrix of common factor scores,
- \( \Lambda \) is the \( N \times K \) loading matrix,
- \( U \) is the idiosyncratic residual matrix,
- \( K \) is the number of retained principal components.

### 2.5 Covariance matrix and eigen decomposition

PCA is performed on the covariance structure of the standardized block:
""")

        st.latex(r"""
\Sigma_Z = \frac{1}{T-1} Z'Z
""")

        st.markdown(r"""
The principal components are obtained from the eigenvalue problem:
""")

        st.latex(r"""
\Sigma_Z v_j = \lambda_j v_j
""")

        st.markdown(r"""
where:

- \( \lambda_j \) is the \( j \)-th eigenvalue,
- \( v_j \) is the corresponding eigenvector.

The eigenvalues measure explained variance:
""")

        st.latex(r"""
\text{Explained Variance Ratio of Component } j
=
\frac{\lambda_j}{\sum_{k=1}^{N}\lambda_k}
""")

        st.markdown(r"""
### 2.6 Principal component scores

For each retained component, the score is:
""")

        st.latex(r"""
PC_{jt} = Z_t v_j
""")

        st.markdown(r"""
where \( Z_t \) is the row vector of standardized variables at time \( t \).

### 2.7 Rotation

To improve interpretability, the loading matrix is rotated using Varimax rotation.
The idea is to find an orthogonal rotation matrix \( R \) such that:
""")

        st.latex(r"""
\Lambda^{*} = \Lambda R
""")

        st.markdown(r"""
and the rotated scores become:
""")

        st.latex(r"""
F^{*} = FR
""")

        st.markdown(r"""
The first rotated component for each block is then used as the operational factor index.

### 2.8 Forecasting equation

After factor construction, the forecasting layer uses lagged inflation, lagged factors, and direct controls.

A generic representation is:
""")

        st.latex(r"""
\hat{\pi}_{t+h}
=
f\left(
\pi_t,\pi_{t-1},\pi_{t-2},
F_t,F_{t-1},F_{t-2},
X_t
\right)
""")

        st.markdown(r"""
where:

- \( \pi_t \) is the target inflation series,
- \( F_t \) denotes the extracted factor vector,
- \( X_t \) denotes direct macro-financial controls,
- \( f(\cdot) \) depends on the chosen forecasting model.

### 2.9 Residual uncertainty bands

For operational fan charts, the application uses residual-based uncertainty bands.
If \( \hat{\sigma}_{\varepsilon} \) is the residual standard deviation from the selected model, the forecast intervals are approximated by:
""")

        st.latex(r"""
\hat{\pi}_{t+h}^{68\%}
=
\hat{\pi}_{t+h} \pm \hat{\sigma}_{\varepsilon}
""")

        st.latex(r"""
\hat{\pi}_{t+h}^{95\%}
=
\hat{\pi}_{t+h} \pm 1.96\hat{\sigma}_{\varepsilon}
""")

        st.markdown(r"""
These are operational uncertainty bands rather than full structural density forecasts.
""")

    with st.expander("3. Step-by-step procedure", expanded=False):
        st.markdown(r"""
### Step 1: Data ingestion
The application reads a user-uploaded or local CSV file containing:
- a `date` column,
- a `cpi` column,
- and optional macro-financial drivers.

### Step 2: Indicator construction
If not already supplied, the system derives:
- monthly inflation,
- annual inflation,
- quarterly inflation,
- annualised quarterly inflation,
- core inflation measures.

### Step 3: Proxy construction
Where some economic drivers are missing, the application constructs proxy variables such as:

- policy rate proxy,
- fiscal impulse proxy,
- imported inflation proxy,
- inflation expectations proxy,
- wage pressure proxy,
- output gap proxy,
- inflation diffusion proxy.

These proxies are reduced-form operational approximations, used to preserve analytical continuity when complete source data are unavailable.

### Step 4: Data cleaning and preparation
The system then:
- converts variables to numeric format,
- handles missing values through interpolation and imputation,
- winsorizes extreme values where relevant,
- and creates lag structures.

If \( x_t \) is a variable, the lag operator is:
""")

        st.latex(r"""
L^k x_t = x_{t-k}
""")

        st.markdown(r"""
Thus, target lags and factor lags are included as part of the predictive feature set.

### Step 5: Factor extraction
Variables are grouped into interpretable economic blocks:

- domestic liquidity,
- external cost,
- inflation momentum,
- demand/fiscal conditions.

Within each block, variables are standardized and passed through PCA.
The first rotated principal component is retained as the block summary index.

### Step 6: Forecast model estimation
The model estimates multiple candidate forecasting models and compares them using out-of-sample RMSE.

### Step 7: Regime classification
Annual inflation is mapped to policy-relevant categories:
- Below Lower Band
- Within SADC Band
- Moderately Above Band
- Far Above Band

### Step 8: Scenario analysis
User-imposed shocks are applied to selected drivers, factors are recomputed, and the model generates a revised nowcast and forecast path.

In compact form, a scenario shock may be represented as:
""")

        st.latex(r"""
X_t^{scenario} = X_t + s
""")

        st.markdown(r"""
where \( s \) is the vector of user-specified shocks.
""")

    with st.expander("4. Model assumptions", expanded=False):
        st.markdown(r"""
The framework relies on the following main assumptions:

1. **CPI is a valid anchor variable** for inflation measurement.
2. **Proxy variables are economically informative** when direct data are unavailable.
3. **Latent macroeconomic conditions can be summarized by a small number of factors**.
4. **Relationships in the training sample are sufficiently stable** to support short-horizon forecasting.
5. **Lagged inflation and macro drivers contain predictive information** about near-term inflation.
6. **Scenario shocks are comparative policy simulations**, not structural causal certainties.
7. **Confidence bands are residual-based approximations**, not full density forecasts from a structural DSGE or Bayesian system.

### Statistical assumptions used operationally

Although the app is not a fully structural econometric system, it implicitly assumes:

- the training sample contains usable information about future behavior,
- the transformed feature matrix is sufficiently informative,
- the signal-to-noise ratio is not negligible,
- the forecasting relation is locally stable over the operational horizon,
- and the scenario shocks are interpreted conditionally rather than causally.

### Interpretation of assumptions

The model should therefore be seen as:

- a disciplined empirical forecasting system,
- a policy-support tool,
- a scenario engine,
- and a comparative analytics framework,

rather than as a full theory-consistent structural model of the Zimbabwean economy.
""")

    with st.expander("5. Factor block interpretation", expanded=False):
        st.markdown(r"""
### Overview

The factor layer compresses several observed macroeconomic variables into a smaller number of economically interpretable indices.
Each factor is intended to summarize a broad inflation driver rather than a single observed series.

In general, a factor for block \( b \) can be written as:
""")

        st.latex(r"""
F_{b,t} = \sum_{j=1}^{N_b} \omega_{bj} z_{j,t}
""")

        st.markdown(r"""
where:

- \( F_{b,t} \) is the factor score for block \( b \) at time \( t \),
- \( z_{j,t} \) is the standardized value of variable \( j \),
- \( \omega_{bj} \) is the rotated loading weight attached to variable \( j \),
- \( N_b \) is the number of variables in block \( b \).

Thus, each factor is a weighted composite index of related macroeconomic signals.

### Domestic Liquidity Factor

This factor is intended to summarize underlying monetary and payments-system liquidity conditions.
It typically combines variables such as:

- broad money growth,
- reserve money growth,
- RTGS growth,
- mobile money growth,
- POS growth,
- private sector credit growth,
- policy rate conditions.

Economically, this factor reflects the extent to which monetary expansion or liquidity creation may be exerting pressure on aggregate demand and prices.

A stylized interpretation is:
""")

        st.latex(r"""
F^{DL}_t \uparrow
\quad \Rightarrow \quad
\text{stronger underlying liquidity pressure}
""")

        st.markdown(r"""
### External Cost Factor

This factor captures imported and pass-through pressures from:

- exchange rate changes,
- exchange rate volatility,
- fuel price changes,
- producer/imported price proxies,
- external cost shocks.

A stylized pass-through interpretation is:
""")

        st.latex(r"""
\frac{\partial \pi_t}{\partial F^{EC}_t} > 0
""")

        st.markdown(r"""
meaning that higher external cost pressure is generally associated with higher inflation.

### Inflation Momentum Factor

This factor captures inflation persistence and internal propagation mechanisms, including:

- core monthly inflation,
- core annual inflation,
- expectations proxies,
- diffusion measures,
- wage growth pressure.

This factor is intended to summarize the extent to which inflation has become embedded in the system.

A persistence-style interpretation is:
""")

        st.latex(r"""
\pi_t = \rho \pi_{t-1} + \varepsilon_t,
\qquad
0 < \rho < 1
""")

        st.markdown(r"""
The inflation momentum factor can be viewed as an empirical summary of this persistence structure when multiple persistence-related indicators are combined.

### Demand/Fiscal Factor

This factor captures domestic demand and fiscal impulses through variables such as:

- output gap proxies,
- fiscal impulse measures,
- price-level demand indicators.

A positive movement in this factor suggests stronger internal demand pressure or fiscal support that may feed into inflation.

### Why factors matter

The factor approach is useful because observed macroeconomic variables are often:

- noisy,
- correlated,
- partially redundant,
- and measured with error.

By extracting common variation, the model attempts to isolate underlying inflationary conditions more efficiently than by treating every raw variable independently.
""")

    with st.expander("6. Forecasting models used", expanded=False):
        st.markdown(r"""
### Overview

The forecasting layer is designed as a **multi-model prediction system** rather than a single-model framework.
This is important because inflation dynamics are rarely governed by one stable functional form.
In some periods, inflation may behave approximately linearly and persistently, while in other periods it may respond in strongly nonlinear ways to exchange-rate shocks, liquidity surges, administered price changes, or expectations shifts.

The forecasting problem is written generally as:
""")

        st.latex(r"""
\hat{\pi}_{t+h} = f(Z_t)
""")

        st.markdown(r"""
where \( Z_t \) is the feature vector containing lagged inflation, factor scores, lagged factors, and direct macroeconomic controls.

A more explicit feature representation is:
""")

        st.latex(r"""
Z_t =
\left[
\pi_t,\pi_{t-1},\pi_{t-2},
F_t,F_{t-1},F_{t-2},
X_t
\right]
""")

        st.markdown(r"""
where:

- \( \pi_t \) is the inflation target series,
- \( F_t \) denotes factor scores,
- \( X_t \) denotes direct controls such as exchange-rate changes, fuel changes, policy rates, and fiscal proxies.

---

### 6.1 Elastic Net

Elastic Net is used as a **regularized linear benchmark**.
It is useful when predictors are numerous and correlated.

The linear forecasting equation is:
""")

        st.latex(r"""
\hat{\pi}_{t+h} = \beta_0 + \sum_{j=1}^{p}\beta_j Z_{jt}
""")

        st.markdown(r"""
Elastic Net estimates coefficients by solving:
""")

        st.latex(r"""
\min_{\beta_0,\beta}
\left\{
\sum_{t=1}^{T}
\left(\pi_{t+h} - \beta_0 - \sum_{j=1}^{p}\beta_j Z_{jt}\right)^2
+
\lambda
\left[
(1-\alpha)\frac{1}{2}\sum_{j=1}^{p}\beta_j^2
+
\alpha\sum_{j=1}^{p}|\beta_j|
\right]
\right\}
""")

        st.markdown(r"""
where:

- \( \lambda \) is the penalty strength,
- \( \alpha \in [0,1] \) balances Ridge and LASSO shrinkage.

This is valuable because the feature matrix may contain multicollinearity across lags, factors, and proxies.

#### Contribution interpretation

Because Elastic Net is coefficient-based, the contribution of predictor \( j \) to the fitted value may be expressed as:
""")

        st.latex(r"""
C_{j,t} = \beta_j Z_{j,t}
""")

        st.markdown(r"""
and the total prediction is:
""")

        st.latex(r"""
\hat{\pi}_{t+h} = \beta_0 + \sum_{j=1}^{p} C_{j,t}
""")

        st.markdown(r"""
This makes Elastic Net especially useful for interpretation and communication.

---

### 6.2 Random Forest

Random Forest is a nonlinear ensemble model made of many decision trees.

A single tree partitions the predictor space into regions \( R_1,\ldots,R_M \) and assigns a fitted value to each region:
""")

        st.latex(r"""
\hat{\pi}_{t+h} = \sum_{m=1}^{M} c_m \mathbf{1}(Z_t \in R_m)
""")

        st.markdown(r"""
Random Forest aggregates many such trees:
""")

        st.latex(r"""
\hat{\pi}_{t+h}^{RF} = \frac{1}{B}\sum_{b=1}^{B} T_b(Z_t)
""")

        st.markdown(r"""
where:

- \( B \) is the number of trees,
- \( T_b(\cdot) \) is the prediction from tree \( b \).

This is useful when:
- pass-through is nonlinear,
- variable interactions matter,
- inflation responds differently across states of the economy.

---

### 6.3 Support Vector Regression (SVR)

SVR is a margin-based nonlinear regression method.
It searches for a function that is as flat as possible while tolerating errors within an \(\varepsilon\)-insensitive band.

The optimization problem is:
""")

        st.latex(r"""
\min_{w,b,\xi_t,\xi_t^*}
\frac{1}{2}\|w\|^2 + C\sum_{t=1}^{T}(\xi_t + \xi_t^*)
""")

        st.markdown(r"""
subject to:
""")

        st.latex(r"""
\pi_{t+h} - w'Z_t - b \leq \varepsilon + \xi_t
""")

        st.latex(r"""
w'Z_t + b - \pi_{t+h} \leq \varepsilon + \xi_t^*
""")

        st.latex(r"""
\xi_t,\xi_t^* \geq 0
""")

        st.markdown(r"""
Under a nonlinear kernel representation:
""")

        st.latex(r"""
f(Z_t) = \sum_{i=1}^{T}(\alpha_i - \alpha_i^*)K(Z_i, Z_t) + b
""")

        st.markdown(r"""
For the radial basis function kernel:
""")

        st.latex(r"""
K(Z_i,Z_t) = \exp\left(-\gamma \|Z_i - Z_t\|^2\right)
""")

        st.markdown(r"""
SVR is useful when the inflation process is nonlinear but the analyst still wants a smooth and controlled prediction function.

---

### 6.4 XGBoost

XGBoost is a gradient boosting algorithm that builds predictions iteratively:
""")

        st.latex(r"""
\hat{\pi}_{t+h}^{(m)} = \hat{\pi}_{t+h}^{(m-1)} + \eta f_m(Z_t)
""")

        st.markdown(r"""
where:

- \( f_m(Z_t) \) is the new tree added at iteration \( m \),
- \( \eta \) is the learning rate.

The regularized objective is:
""")

        st.latex(r"""
\mathcal{L}^{(m)}
=
\sum_{t=1}^{T}
l\left(\pi_{t+h}, \hat{\pi}_{t+h}^{(m-1)} + f_m(Z_t)\right)
+
\Omega(f_m)
""")

        st.markdown(r"""
with a typical tree penalty:
""")

        st.latex(r"""
\Omega(f) = \gamma J + \frac{1}{2}\lambda \sum_{j=1}^{J} w_j^2
""")

        st.markdown(r"""
This model is powerful when the inflation process contains rich interaction structures and threshold effects.

---

### 6.5 Model comparison and selection

All forecasting models are evaluated on a holdout sample.

#### Root Mean Squared Error
""")

        st.latex(r"""
RMSE = \sqrt{\frac{1}{N}\sum_{t=1}^{N}(\pi_t - \hat{\pi}_t)^2}
""")

        st.markdown(r"""
#### Mean Absolute Error
""")

        st.latex(r"""
MAE = \frac{1}{N}\sum_{t=1}^{N}|\pi_t - \hat{\pi}_t|
""")

        st.markdown(r"""
#### Mean Absolute Percentage Error
""")

        st.latex(r"""
MAPE = \frac{100}{N}\sum_{t=1}^{N}\left|\frac{\pi_t - \hat{\pi}_t}{\pi_t}\right|
""")

        st.markdown(r"""
#### Coefficient of Determination
""")

        st.latex(r"""
R^2 = 1 - \frac{\sum_{t=1}^{N}(\pi_t - \hat{\pi}_t)^2}{\sum_{t=1}^{N}(\pi_t - \bar{\pi})^2}
""")

        st.markdown(r"""
The best model is selected primarily using **out-of-sample RMSE**.

### 6.6 Why a multi-model system is preferable

A multi-model system is preferable because inflation forecasting is subject to:

- model uncertainty,
- structural change,
- nonlinear shock transmission,
- changing policy environments,
- evolving expectations formation.

No single forecasting technology dominates in all macroeconomic states.
The multi-model design therefore increases robustness and comparative analytical value.
""")

    with st.expander("7. Regime classification rules", expanded=False):
        st.markdown(r"""
Annual inflation is classified as follows:
""")

        st.latex(r"""
\text{Regime}_t =
\begin{cases}
\text{Below Lower Band}, & \pi^y_t < 3 \\
\text{Within SADC Band}, & 3 \leq \pi^y_t \leq 7 \\
\text{Moderately Above Band}, & 7 < \pi^y_t \leq 15 \\
\text{Far Above Band}, & \pi^y_t > 15
\end{cases}
""")

        st.markdown(r"""
This maps the continuous inflation series into discrete policy communication states.

In classification notation, the regime model can be viewed as learning:
""")

        st.latex(r"""
P(R_t = r \mid Z_t)
""")

        st.markdown(r"""
for each regime \( r \), where \( Z_t \) is the predictor vector.
The classifier then assigns the most likely regime:
""")

        st.latex(r"""
\hat{R}_t = \arg\max_r P(R_t = r \mid Z_t)
""")

    with st.expander("8. Limitations and cautions", expanded=False):
        st.markdown(r"""
This application is a powerful operational forecasting tool, but users should note:

- It is **not a full structural macroeconometric model**.
- It does **not impose explicit economic identification restrictions**.
- It depends on the **quality and continuity of the input data**.
- Scenario results should be interpreted as **conditional simulations**, not exact policy outcomes.
- Residual-based uncertainty bands may **understate tail risks** during structural breaks.

### Additional technical cautions

1. **Proxy dependence**  
   Some inputs are proxy constructs rather than directly observed policy or market variables.

2. **Reduced-form interpretation**  
   The application is fundamentally reduced-form and predictive rather than causally identified.

3. **Sample dependence**  
   Model performance depends on the training sample and may change when structural relationships shift.

4. **Non-structural uncertainty bands**  
   Forecast bands are residual-based approximations rather than model-consistent probability distributions.

5. **Conditional scenario analysis**  
   Scenario outputs answer “what would the forecast look like if these inputs changed,” not “what will definitely happen.”
""")

    with st.expander("9. Recommended interpretation for policy users", expanded=False):
        st.markdown(r"""
Policy users should interpret outputs as follows:

### Nowcast
The model’s estimate of inflation under current observed conditions.

### Forecast path
The expected near-term inflation trajectory conditional on the maintained predictor structure and chosen scenario.

### Factors
Compressed summaries of broad inflation drivers rather than directly observed policy instruments.

### Regime classification
A policy communication device showing where inflation lies relative to target thresholds.

### Scenario outputs
Comparative conditional stress tests under alternative assumptions.

### Practical central-bank interpretation

A useful way to read the model is:

1. **Look at the current nowcast** to assess present inflation conditions.
2. **Look at factor movements** to understand the broad sources of pressure.
3. **Look at the forecast path** to assess near-term persistence and direction.
4. **Look at the regime classification** to position inflation relative to policy thresholds.
5. **Look at the scenario results** to compare policy alternatives or shock environments.

Thus, the application should be used as a structured policy-support system, not as a mechanical replacement for expert macroeconomic judgment.
""")

    with st.expander("10. Scenario engine mathematics", expanded=False):
        st.markdown(r"""
### Overview

The scenario engine is a **conditional forecasting mechanism**.
It asks the following policy-relevant question:

**If selected macroeconomic drivers were shifted away from their baseline values, how would those shifts propagate through the factor system and into the inflation forecast?**

The logic of the scenario engine is:
""")

        st.latex(r"""
s_t
\;\longrightarrow\;
X_t^{(s)}
\;\longrightarrow\;
Z_t^{(s)}
\;\longrightarrow\;
F_t^{(s)}
\;\longrightarrow\;
\widetilde{Z}_t^{(s)}
\;\longrightarrow\;
\hat{\pi}_{t+h}^{(s)}
""")

        st.markdown(r"""
where:

- \( s_t \) is the vector of imposed scenario shocks,
- \( X_t^{(s)} \) is the shocked raw macroeconomic driver vector,
- \( Z_t^{(s)} \) is the shocked standardized variable vector,
- \( F_t^{(s)} \) is the shocked factor vector,
- \( \widetilde{Z}_t^{(s)} \) is the final model feature vector under the scenario,
- \( \hat{\pi}_{t+h}^{(s)} \) is the scenario-conditioned inflation forecast.

The propagation mechanism therefore moves through **three distinct layers**:

1. **Observed variables**
2. **Latent factors**
3. **Forecast model**

---

### 10.1 Baseline macroeconomic state

Let the observed macro-financial state at time \( t \) be:
""")

        st.latex(r"""
X_t =
\begin{bmatrix}
x_{1t} \\
x_{2t} \\
\vdots \\
x_{nt}
\end{bmatrix}
""")

        st.markdown(r"""
These may include variables such as:

- exchange-rate change,
- fuel-price change,
- money growth,
- reserve-money growth,
- fiscal impulse proxy,
- policy rate,
- inflation expectations proxy,
- imported inflation proxy,
- output gap proxy.

This vector \( X_t \) is the **baseline latest macroeconomic position** used by the application.

---

### 10.2 User-imposed shock vector

Suppose the user chooses a policy or shock scenario.
That scenario is represented by a shock vector:
""")

        st.latex(r"""
s_t =
\begin{bmatrix}
s_{1t} \\
s_{2t} \\
\vdots \\
s_{nt}
\end{bmatrix}
""")

        st.markdown(r"""
Each element \( s_{it} \) is the adjustment imposed on variable \( i \).

Examples:

- \( s_{\text{policy rate},t} > 0 \): tighter monetary stance
- \( s_{\text{m3 growth},t} < 0 \): monetary contraction
- \( s_{\text{exchange-rate change},t} > 0 \): depreciation shock
- \( s_{\text{fuel change},t} > 0 \): fuel price shock
- \( s_{\text{fiscal impulse},t} < 0 \): fiscal tightening

The shocked observed state becomes:
""")

        st.latex(r"""
X_t^{(s)} = X_t + s_t
""")

        st.markdown(r"""
Element by element:
""")

        st.latex(r"""
x_{it}^{(s)} = x_{it} + s_{it}
""")

        st.markdown(r"""
This is the first stage of propagation.
The scenario engine therefore begins by replacing the baseline latest observations with their shocked equivalents.

---

### 10.3 Standardization under the scenario

The factor engine does not work directly with raw variables.
Instead, it works with standardized variables.

For each variable \( x_j \), standardization is:
""")

        st.latex(r"""
z_{jt} = \frac{x_{jt} - \mu_j}{\sigma_j}
""")

        st.markdown(r"""
where:

- \( \mu_j \) is the historical mean of variable \( j \),
- \( \sigma_j \) is the historical standard deviation of variable \( j \).

Under the scenario, the standardized variable becomes:
""")

        st.latex(r"""
z_{jt}^{(s)} = \frac{x_{jt}^{(s)} - \mu_j}{\sigma_j}
= \frac{x_{jt} + s_{jt} - \mu_j}{\sigma_j}
""")

        st.markdown(r"""
This can be decomposed as:
""")

        st.latex(r"""
z_{jt}^{(s)}
=
\frac{x_{jt} - \mu_j}{\sigma_j}
+
\frac{s_{jt}}{\sigma_j}
=
z_{jt} + \frac{s_{jt}}{\sigma_j}
""")

        st.markdown(r"""
This is a very important result:

### Interpretation

A raw shock \( s_{jt} \) does not enter the factor system in raw units.
It enters in **standardized units**, scaled by the historical variability of that variable.

Therefore:

- a given shock has a larger factor impact if \( \sigma_j \) is small,
- a given shock has a smaller factor impact if \( \sigma_j \) is large.

So the model interprets shocks **relative to the historical volatility** of the affected variable.

---

### 10.4 Shock propagation into the factor block

Suppose factor block \( b \) contains \( N_b \) variables.
Let the rotated loading vector for that block be:
""")

        st.latex(r"""
\omega_b =
\begin{bmatrix}
\omega_{b1} \\
\omega_{b2} \\
\vdots \\
\omega_{bN_b}
\end{bmatrix}
""")

        st.markdown(r"""
and let the shocked standardized block vector be:
""")

        st.latex(r"""
z_{b,t}^{(s)} =
\begin{bmatrix}
z_{1t}^{(s)} \\
z_{2t}^{(s)} \\
\vdots \\
z_{N_b t}^{(s)}
\end{bmatrix}
""")

        st.markdown(r"""
Then the scenario factor score for block \( b \) is:
""")

        st.latex(r"""
F_{b,t}^{(s)} = \omega_b' z_{b,t}^{(s)}
""")

        st.markdown(r"""
or equivalently:
""")

        st.latex(r"""
F_{b,t}^{(s)} = \sum_{j=1}^{N_b}\omega_{bj} z_{jt}^{(s)}
""")

        st.markdown(r"""
Substituting the shocked standardized variables:
""")

        st.latex(r"""
F_{b,t}^{(s)}
=
\sum_{j=1}^{N_b}
\omega_{bj}
\left(
z_{jt} + \frac{s_{jt}}{\sigma_j}
\right)
""")

        st.markdown(r"""
Expanding:
""")

        st.latex(r"""
F_{b,t}^{(s)}
=
\sum_{j=1}^{N_b}\omega_{bj} z_{jt}
+
\sum_{j=1}^{N_b}\omega_{bj}\frac{s_{jt}}{\sigma_j}
""")

        st.markdown(r"""
Hence:
""")

        st.latex(r"""
F_{b,t}^{(s)} = F_{b,t} + \Delta F_{b,t}
""")

        st.latex(r"""
\Delta F_{b,t}
=
\sum_{j=1}^{N_b}\omega_{bj}\frac{s_{jt}}{\sigma_j}
""")

        st.markdown(r"""
This is the key propagation equation.

### Interpretation of the factor shock component

The change in factor \( b \) caused by the scenario is:
""")

        st.latex(r"""
\Delta F_{b,t}
=
\sum_{j=1}^{N_b}\omega_{bj}\frac{s_{jt}}{\sigma_j}
""")

        st.markdown(r"""
So factor transmission depends on three things:

1. **Shock size** \( s_{jt} \)  
2. **Variable scaling** \( \sigma_j \)  
3. **Factor loading weight** \( \omega_{bj} \)

This means a shock affects the factor more strongly when:

- the shock is large,
- the shocked variable is historically stable,
- the variable has a high absolute rotated loading in that factor block.

---

### 10.5 Interpretation by factor block

Suppose the system has the following factors:

- domestic liquidity factor \( F_t^{DL} \)
- external cost factor \( F_t^{EC} \)
- inflation momentum factor \( F_t^{IM} \)
- demand/fiscal factor \( F_t^{DF} \)

Then under the scenario, each factor becomes:
""")

        st.latex(r"""
F_t^{DL,(s)} = F_t^{DL} + \Delta F_t^{DL}
""")

        st.latex(r"""
F_t^{EC,(s)} = F_t^{EC} + \Delta F_t^{EC}
""")

        st.latex(r"""
F_t^{IM,(s)} = F_t^{IM} + \Delta F_t^{IM}
""")

        st.latex(r"""
F_t^{DF,(s)} = F_t^{DF} + \Delta F_t^{DF}
""")

        st.markdown(r"""
For example:

- a tightening scenario may reduce \( F_t^{DL} \),
- a depreciation shock may raise \( F_t^{EC} \),
- an expectations shock may raise \( F_t^{IM} \),
- a fiscal expansion scenario may raise \( F_t^{DF} \).

Thus, the scenario engine works by transforming raw shocks into **latent economic pressure indices**.

---

### 10.6 Smoothing and operational factor updating

In the application, factor estimates are operationally smoothed using a moving average window.
If the smoothing window is of length \( q \), then the operational scenario factor may be written as:
""")

        st.latex(r"""
\bar{F}_{b,t}^{(s)} = \frac{1}{q}\sum_{r=0}^{q-1} F_{b,t-r}^{(s)}
""")

        st.markdown(r"""
This means the final factor used by the forecast engine may not be the raw one-period shocked factor alone,
but a smoothed version combining recent factor history and the current shocked factor.

This matters because it dampens excessively volatile one-off shocks and reflects the model’s attempt to represent macro conditions more smoothly.

---

### 10.7 Construction of the final scenario feature vector

Once the shocked factors are computed, the full predictor vector used by the forecasting model is assembled.

The baseline predictor vector is:
""")

        st.latex(r"""
\widetilde{Z}_t =
\left[
\pi_t,\pi_{t-1},\pi_{t-2},
F_t,F_{t-1},F_{t-2},
X_t
\right]
""")

        st.markdown(r"""
Under the scenario, it becomes:
""")

        st.latex(r"""
\widetilde{Z}_t^{(s)} =
\left[
\pi_t,\pi_{t-1},\pi_{t-2},
F_t^{(s)},F_{t-1},F_{t-2},
X_t^{(s)}
\right]
""")

        st.markdown(r"""
In the one-step nowcast, the current-period factors and current-period shocked controls are updated, while lagged terms are carried forward from the latest observed history.

Thus, the model combines:

- unchanged historical lags,
- updated current shocked controls,
- updated current shocked factors.

---

### 10.8 Shock propagation into a linear forecast model

To see the mechanics transparently, consider the Elastic Net or any linear forecasting equation:
""")

        st.latex(r"""
\hat{\pi}_{t+h}
=
\beta_0
+
\beta_{\pi,0}\pi_t
+
\beta_{\pi,1}\pi_{t-1}
+
\beta_{\pi,2}\pi_{t-2}
+
\sum_{b}\beta_{F_b}F_{b,t}
+
\sum_{k}\beta_{X_k}x_{k,t}
""")

        st.markdown(r"""
Under the scenario:
""")

        st.latex(r"""
\hat{\pi}_{t+h}^{(s)}
=
\beta_0
+
\beta_{\pi,0}\pi_t
+
\beta_{\pi,1}\pi_{t-1}
+
\beta_{\pi,2}\pi_{t-2}
+
\sum_{b}\beta_{F_b}F_{b,t}^{(s)}
+
\sum_{k}\beta_{X_k}x_{k,t}^{(s)}
""")

        st.markdown(r"""
Subtracting the baseline forecast gives the forecast effect of the scenario:
""")

        st.latex(r"""
\Delta \hat{\pi}_{t+h}
=
\hat{\pi}_{t+h}^{(s)} - \hat{\pi}_{t+h}
=
\sum_b \beta_{F_b}\Delta F_{b,t}
+
\sum_k \beta_{X_k}s_{k,t}
""")

        st.markdown(r"""
Substituting the factor shock term:
""")

        st.latex(r"""
\Delta \hat{\pi}_{t+h}
=
\sum_b \beta_{F_b}
\left(
\sum_{j=1}^{N_b}\omega_{bj}\frac{s_{jt}}{\sigma_j}
\right)
+
\sum_k \beta_{X_k}s_{k,t}
""")

        st.markdown(r"""
This is the full linear shock-transmission equation.

### Interpretation

A scenario affects the forecast through **two channels**:

#### Direct channel
""")

        st.latex(r"""
\sum_k \beta_{X_k}s_{k,t}
""")

        st.markdown(r"""
This is the effect of shocked raw macro variables entering the forecast equation directly.

#### Indirect factor channel
""")

        st.latex(r"""
\sum_b \beta_{F_b}
\left(
\sum_{j=1}^{N_b}\omega_{bj}\frac{s_{jt}}{\sigma_j}
\right)
""")

        st.markdown(r"""
This is the effect of the same shocks entering indirectly by changing the factor scores.

This is exactly the propagation mechanism:

**shock → standardized variable → factor score → forecast**

---

### 10.9 Matrix form of the same propagation

Let:

- \( s_t \) be the vector of shocks,
- \( D^{-1} \) be the diagonal matrix with entries \( 1/\sigma_j \),
- \( W \) be the matrix of factor loading weights,
- \( \beta_F \) be the vector of coefficients on the factors,
- \( \beta_X \) be the vector of coefficients on the direct controls.

Then:
""")

        st.latex(r"""
\Delta F_t = W' D^{-1} s_t
""")

        st.markdown(r"""
and the forecast impact becomes:
""")

        st.latex(r"""
\Delta \hat{\pi}_{t+h}
=
\beta_F' \Delta F_t + \beta_X' s_t
""")

        st.latex(r"""
=
\beta_F' W' D^{-1} s_t + \beta_X' s_t
""")

        st.markdown(r"""
Therefore:
""")

        st.latex(r"""
\Delta \hat{\pi}_{t+h}
=
\left(\beta_F' W' D^{-1} + \beta_X' \right)s_t
""")

        st.markdown(r"""
This equation shows the entire scenario engine as a single shock-to-forecast mapping.

It says that the effect of a scenario is a weighted linear combination of:

- direct coefficients on shocked observed variables,
- indirect coefficients operating through the factor structure.

Even when the final forecasting model is nonlinear, this linear representation is still extremely useful for interpretation because it clarifies the transmission logic.

---

### 10.10 Scenario propagation in nonlinear models

For nonlinear models such as Random Forest, SVR, and XGBoost, the exact analytic derivative is generally not represented by a single fixed coefficient vector.
Instead, the model computes:
""")

        st.latex(r"""
\hat{\pi}_{t+h}^{(s)} = f\left(\widetilde{Z}_t^{(s)}\right)
""")

        st.markdown(r"""
and the scenario effect is:
""")

        st.latex(r"""
\Delta \hat{\pi}_{t+h}
=
f\left(\widetilde{Z}_t^{(s)}\right)
-
f\left(\widetilde{Z}_t\right)
""")

        st.markdown(r"""
For small shocks, a first-order local approximation may be written as:
""")

        st.latex(r"""
\Delta \hat{\pi}_{t+h}
\approx
\nabla f(\widetilde{Z}_t)' \Delta \widetilde{Z}_t
""")

        st.markdown(r"""
where \( \nabla f(\widetilde{Z}_t) \) is the gradient of the nonlinear prediction surface evaluated at the baseline point.

This means that even in nonlinear models, the same basic propagation logic still applies:

1. shock the observed variables,
2. recompute factors,
3. rebuild the feature vector,
4. pass the new feature vector through the nonlinear model.

The difference is that the final mapping \( f(\cdot) \) is nonlinear and state-dependent.

---

### 10.11 Multi-step forecast propagation

For a multi-step forecast path, the scenario engine operates recursively.

At horizon \( h=1 \):
""")

        st.latex(r"""
\hat{\pi}_{t+1}^{(s)} = f\left(\widetilde{Z}_t^{(s)}\right)
""")

        st.markdown(r"""
At horizon \( h=2 \), the one-step-ahead forecast may itself become an input through lag updating:
""")

        st.latex(r"""
\hat{\pi}_{t+2}^{(s)}
=
f\left(
\hat{\pi}_{t+1}^{(s)},
\pi_t,
\pi_{t-1},
F_{t+1}^{(s)},
F_t^{(s)},
F_{t-1},
X_{t+1}^{(s)}
\right)
""")

        st.markdown(r"""
More generally:
""")

        st.latex(r"""
\hat{\pi}_{t+h}^{(s)} = f\left(\widetilde{Z}_{t+h-1}^{(s)}\right)
\qquad h = 1,2,\dots,H
""")

        st.markdown(r"""
Thus, a scenario can have:

- an **immediate effect** at horizon 1,
- a **propagated effect** through lagged forecast updating at later horizons.

This is why policy shocks may generate persistent forecast-path changes rather than one-period jumps only.

---

### 10.12 Example: contractionary monetary scenario

Suppose the scenario is:

- higher policy rate,
- lower money growth,
- lower reserve money growth,
- slightly stronger exchange rate,
- reduced inflation expectations.

Then:

1. raw shocked variables are created,
2. the domestic liquidity factor typically falls,
3. the inflation momentum factor may also fall,
4. the forecast model receives lower factor inputs and tighter direct controls,
5. the inflation nowcast and forecast path decline relative to baseline.

In stylized notation:
""")

        st.latex(r"""
s_t^{hawkish}
\Rightarrow
\Delta F_t^{DL} < 0,
\quad
\Delta F_t^{IM} < 0
\Rightarrow
\Delta \hat{\pi}_{t+h} < 0
""")

        st.markdown(r"""
This illustrates how policy scenarios become economically interpretable within the model.

---

### 10.13 Practical interpretation for users

The scenario engine should be interpreted as a **conditional transmission system**.

It does **not** say:

- the economy will automatically move to those shocked values,
- all endogenous adjustments are fully modeled,
- the resulting forecast is a structural causal certainty.

It **does** say:

- given the estimated historical relationships in the data,
- and given the factor structure extracted from that data,
- if the selected drivers were changed by the scenario,
- then the model would map those changes into the reported nowcast and forecast path.

So the scenario engine is best understood as:

- a **policy stress-testing tool**,
- a **conditional projection framework**,
- and a **transparent shock-propagation system** linking macroeconomic assumptions to inflation outcomes.
""")
st.markdown("---")

with tab7:
    st.title("Disclaimer and Ownership Notice")

    st.markdown(r"""
### Disclaimer

This application has been developed as a technical analytical and policy-support tool for inflation nowcasting, forecasting, scenario simulation, and regime assessment.

The outputs generated by this system are model-based estimates derived from historical data, statistical procedures, latent factor extraction, and machine-learning forecasting techniques. Accordingly, the results should be interpreted as **conditional analytical outputs** rather than as official forecasts, binding policy positions, or guaranteed future outcomes.

While due care has been taken in the conceptualization, design, and implementation of the framework, the model remains subject to limitations arising from:

- data quality and continuity,
- the use of proxy variables where direct measures are unavailable,
- structural changes in the economy,
- model uncertainty,
- and the simplifying assumptions underlying factor-based and reduced-form forecasting systems.

Users are therefore advised to apply professional judgment when interpreting outputs, particularly in high-level policy, operational, financial, or strategic contexts.

### Ownership and Intellectual Attribution

This application, including its analytical architecture, factor-engine design, forecasting system, scenario engine, regime-classification structure, and methodological documentation, is the intellectual work of:

**Chirume A.T.**  
**Ph.D\* (DUT, Candidate), MA (Uni of Tsukuba, Japan), MSc, BSc (Hons) Econ (UZ), DSML (MIT)**

### Further Enquiries

For further enquiries, clarification, or technical correspondence, refer to the author:

**Chirume A.T.**  
**atchirume@gmail.com
**Mobile:** +263 773 369 884
""")
st.caption("app.py")
