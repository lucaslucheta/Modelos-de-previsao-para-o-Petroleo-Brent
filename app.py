"""
================================================================================
BRENT QUANT TERMINAL — MODELAGEM ECONOMÉTRICA E PREVISÃO INSTITUCIONAL (BZ=F)
================================================================================
Instalação das dependências:
    pip install streamlit yfinance pandas numpy scikit-learn plotly

Execução do aplicativo:
    streamlit run app.py
================================================================================
"""

import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler
import streamlit as st
import yfinance as yf

try:
    from statsmodels.tsa.stattools import adfuller as _sm_adfuller

    HAS_STATSMODELS = True
except ImportError:
    _sm_adfuller = None
    HAS_STATSMODELS = False

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. CONFIGURAÇÃO DA PÁGINA E ESTILO INSTITUCIONAL
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Brent Quant Terminal | Econometria & Previsão",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.0rem;
            font-weight: 700;
            color: #F8FAFC;
            margin-bottom: 0.15rem;
            letter-spacing: -0.02em;
        }
        .sub-header {
            font-size: 0.95rem;
            color: #94A3B8;
            margin-bottom: 1.2rem;
        }
        .zscore-banner-fair {
            background: linear-gradient(90deg, rgba(16,185,129,0.16) 0%, rgba(15,23,42,0.8) 100%);
            border-left: 5px solid #10B981;
            padding: 14px 18px;
            border-radius: 6px;
            margin-bottom: 1rem;
        }
        .zscore-banner-over {
            background: linear-gradient(90deg, rgba(239,68,68,0.20) 0%, rgba(15,23,42,0.8) 100%);
            border-left: 5px solid #EF4444;
            padding: 14px 18px;
            border-radius: 6px;
            margin-bottom: 1rem;
        }
        .zscore-banner-under {
            background: linear-gradient(90deg, rgba(56,189,248,0.20) 0%, rgba(15,23,42,0.8) 100%);
            border-left: 5px solid #38BDF8;
            padding: 14px 18px;
            border-radius: 6px;
            margin-bottom: 1rem;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(30, 41, 59, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.16);
            padding: 12px 14px;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# 2. METADADOS DAS 9 MOEDAS E CONVENÇÕES DE CÂMBIO REAL
# ------------------------------------------------------------------------------
CURRENCIES: Dict[str, Dict] = {
    "USD": {
        "name": "Dólar Americano (USD)",
        "symbol": "$",
        "tickers": [],
        "decimals": 2,
    },
    "BRL": {
        "name": "Real Brasileiro (BRL)",
        "symbol": "R$",
        "tickers": [("BRL=X", False), ("USDBRL=X", False)],
        "decimals": 2,
    },
    "EUR": {
        "name": "Euro (EUR)",
        "symbol": "€",
        "tickers": [("EURUSD=X", True), ("EUR=X", False)],
        "decimals": 2,
    },
    "GBP": {
        "name": "Libra Esterlina (GBP)",
        "symbol": "£",
        "tickers": [("GBPUSD=X", True), ("GBP=X", False)],
        "decimals": 2,
    },
    "JPY": {
        "name": "Iene Japonês (JPY)",
        "symbol": "¥",
        "tickers": [("JPY=X", False), ("USDJPY=X", False)],
        "decimals": 0,
    },
    "CNY": {
        "name": "Yuan Chinês (CNY)",
        "symbol": "¥",
        "tickers": [("CNY=X", False), ("USDCNY=X", False)],
        "decimals": 2,
    },
    "CHF": {
        "name": "Franco Suíço (CHF)",
        "symbol": "CHF ",
        "tickers": [("CHF=X", False), ("USDCHF=X", False)],
        "decimals": 2,
    },
    "CAD": {
        "name": "Dólar Canadense (CAD)",
        "symbol": "C$",
        "tickers": [("CAD=X", False), ("USDCAD=X", False)],
        "decimals": 2,
    },
    "AUD": {
        "name": "Dólar Australiano (AUD)",
        "symbol": "A$",
        "tickers": [("AUDUSD=X", True), ("AUD=X", False)],
        "decimals": 2,
    },
}


def compute_adf_pvalue(series: pd.Series) -> float:
    """
    Calcula o p-valor do Teste Dickey-Fuller Aumentado (ADF) utilizando statsmodels
    quando disponível ou regressão OLS pura em NumPy com valores críticos de MacKinnon.
    """
    y = np.asarray(series.dropna().values, dtype=np.float64)
    if len(y) < 30:
        return np.nan

    if HAS_STATSMODELS and _sm_adfuller is not None:
        return float(_sm_adfuller(y)[1])

    n = len(y)
    maxlag = max(1, min(int(np.floor(12.0 * (n / 100.0) ** 0.25)), n // 4))
    dy = np.diff(y)
    y_lag1 = y[:-1]

    t_obs = len(dy) - maxlag
    target = dy[maxlag:]
    x_cols = [np.ones(t_obs), y_lag1[maxlag:]]
    for lag in range(1, maxlag + 1):
        x_cols.append(dy[maxlag - lag : len(dy) - lag])
    x_mat = np.column_stack(x_cols)

    beta, _, _, _ = np.linalg.lstsq(x_mat, target, rcond=None)
    resid = target - x_mat @ beta
    s2 = np.sum(resid**2) / max(1, t_obs - x_mat.shape[1])
    cov_beta = s2 * np.linalg.inv(x_mat.T @ x_mat)
    t_stat = float(beta[1] / np.sqrt(cov_beta[1, 1]))

    crit_t = [-4.38, -3.95, -3.43, -3.12, -2.86, -2.57, -2.23, -1.94, -1.56, -1.05, -0.44, 0.25, 1.20]
    crit_p = [0.0001, 0.001, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 0.975, 0.999]
    return float(np.interp(t_stat, crit_t, crit_p))


def format_price(value: float, currency_code: str) -> str:
    """Formata o valor monetário respeitando o símbolo e casas decimais da moeda."""
    meta = CURRENCIES[currency_code]
    dec = meta["decimals"]
    sym = meta["symbol"]
    if np.isnan(value):
        return "N/A"
    return f"{sym}{value:,.{dec}f}"


def _clean_yf_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza colunas MultiIndex e índice de datas do yfinance."""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    df = raw_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _resilient_yf_download(
    ticker: str,
    start_date: Optional[str] = None,
    period: Optional[str] = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Baixa dados do Yahoo Finance com retentativas exponenciais e fallback
    via `yf.Ticker().history()` para evitar falhas por rate-limit (HTTP 429).
    """
    for attempt in range(max_retries):
        try:
            if period is not None:
                raw = yf.download(
                    ticker,
                    period=period,
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            else:
                raw = yf.download(
                    ticker,
                    start=start_date,
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            cleaned = _clean_yf_dataframe(raw)
            if not cleaned.empty and "Close" in cleaned.columns:
                return cleaned
        except Exception:
            pass

        # Fallback via endpoint Ticker().history()
        try:
            tk = yf.Ticker(ticker)
            if period is not None:
                hist = tk.history(period=period, interval="1d", auto_adjust=False)
            else:
                hist = tk.history(start=start_date, interval="1d", auto_adjust=False)
            cleaned_hist = _clean_yf_dataframe(hist)
            if not cleaned_hist.empty and "Close" in cleaned_hist.columns:
                return cleaned_hist
        except Exception:
            pass

        if attempt < max_retries - 1:
            time.sleep(0.7 * (attempt + 1))

    return pd.DataFrame()


# ------------------------------------------------------------------------------
# 3. INGESTÃO DESACOPLADA: BASE USD (BZ=F + EXÓGENAS) E CÂMBIO MULTI-MOEDA
# ------------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_base_usd_market_data() -> Tuple[pd.DataFrame, List[str]]:
    """
    Baixa e armazena em cache a série histórica completa do Brent (BZ=F) em USD
    junto com as variáveis exógenas macro e físicas (DXY, WTI, RBOB, Heating Oil).
    Como o cache não depende da moeda escolhida, alternar moedas nunca refaz
    esse download pesado nem dispara rate-limit no Yahoo Finance.
    """
    df_brent = _resilient_yf_download("BZ=F", period="max", max_retries=4)
    if df_brent.empty:
        raise RuntimeError(
            "O Yahoo Finance não respondeu para BZ=F após múltiplas tentativas. Aguarde alguns segundos e tente novamente."
        )

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df_brent.columns:
            raise ValueError(f"Coluna obrigatória ausente nos dados do Brent: {col}")

    df_brent = df_brent[required_cols].copy()
    df_brent = df_brent[df_brent["Close"] > 0].dropna(subset=["Close"])
    for col in ["Open", "High", "Low"]:
        df_brent[col] = df_brent[col].fillna(df_brent["Close"])
    df_brent["Volume"] = df_brent["Volume"].fillna(0.0)
    df_brent["Close_USD"] = df_brent["Close"].copy()

    start_str = df_brent.index.min().strftime("%Y-%m-%d")
    loaded_exog: List[str] = []

    # 1) Índice Dólar (DXY)
    dxy_df = _resilient_yf_download("DX-Y.NYB", start_date=start_str, max_retries=2)
    if not dxy_df.empty and "Close" in dxy_df.columns:
        s = dxy_df["Close"].replace(0, np.nan).dropna()
        if not s.empty:
            df_brent["DXY"] = s.reindex(df_brent.index).ffill().bfill()
            loaded_exog.append("DXY (DX-Y.NYB)")
    if "DXY" not in df_brent.columns:
        df_brent["DXY"] = 100.0

    # 2) Petróleo WTI (CL=F)
    wti_df = _resilient_yf_download("CL=F", start_date=start_str, max_retries=2)
    if not wti_df.empty and "Close" in wti_df.columns:
        s = wti_df["Close"].replace(0, np.nan).dropna()
        if not s.empty:
            df_brent["WTI_USD"] = s.reindex(df_brent.index).ffill().bfill()
            loaded_exog.append("WTI (CL=F)")
    if "WTI_USD" not in df_brent.columns:
        df_brent["WTI_USD"] = df_brent["Close_USD"]

    # 3) Gasolina RBOB (RB=F) & Óleo de Aquecimento (HO=F)
    rb_df = _resilient_yf_download("RB=F", start_date=start_str, max_retries=2)
    ho_df = _resilient_yf_download("HO=F", start_date=start_str, max_retries=2)
    if (
        not rb_df.empty
        and not ho_df.empty
        and "Close" in rb_df.columns
        and "Close" in ho_df.columns
    ):
        rb_s = rb_df["Close"].replace(0, np.nan).dropna()
        ho_s = ho_df["Close"].replace(0, np.nan).dropna()
        if not rb_s.empty and not ho_s.empty:
            df_brent["RB_USD"] = rb_s.reindex(df_brent.index).ffill().bfill()
            df_brent["HO_USD"] = ho_s.reindex(df_brent.index).ffill().bfill()
            loaded_exog.append("Crack Spread 3-2-1 (RB=F / HO=F)")
    if "RB_USD" not in df_brent.columns:
        df_brent["RB_USD"] = df_brent["Close_USD"] / 42.0
        df_brent["HO_USD"] = df_brent["Close_USD"] / 42.0

    return df_brent, loaded_exog


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fx_series_for_currency(currency_code: str, start_date: str) -> Tuple[pd.Series, str]:
    """
    Baixa de forma resiliente e isolada em cache a série cambial diária para
    converter 1 USD -> Moeda Local (taxa multiplicadora direta).
    """
    if currency_code == "USD":
        return pd.Series(dtype=np.float64), "USD (Base Nativa — Fator 1.0000)"

    meta = CURRENCIES[currency_code]
    for ticker, invert_flag in meta["tickers"]:
        fx_df = _resilient_yf_download(ticker, start_date=start_date, max_retries=3)
        if not fx_df.empty and "Close" in fx_df.columns:
            s = fx_df["Close"].replace(0, np.nan).dropna()
            if len(s) > 30:
                if invert_flag:
                    s = 1.0 / s
                    label = f"{ticker} (1/FX -> {currency_code}/USD)"
                else:
                    label = f"{ticker} (Direto -> {currency_code}/USD)"
                return s, label

    raise RuntimeError(
        f"Não foi possível baixar a série cambial histórica para {currency_code} no Yahoo Finance."
    )


def get_market_data_in_currency(currency_code: str) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Combina os dados base em USD com a série cambial diária exata da moeda
    selecionada, multiplicando cada pregão histórico pelo câmbio daquele dia.
    """
    base_df, loaded_exog = fetch_base_usd_market_data()
    df = base_df.copy()

    if currency_code == "USD":
        df["FX_Rate"] = 1.0
        fx_label = "USD (Base Nativa — Fator 1.0000)"
    else:
        start_str = df.index.min().strftime("%Y-%m-%d")
        fx_series, fx_label = fetch_fx_series_for_currency(currency_code, start_str)
        aligned_fx = fx_series.reindex(df.index).ffill().bfill()
        if aligned_fx.isna().all():
            raise RuntimeError(f"Falha ao alinhar datas de câmbio para {currency_code}.")
        df["FX_Rate"] = aligned_fx.values
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col] * df["FX_Rate"]

    df["Volume_Financeiro"] = df["Close"] * df["Volume"]
    return df, fx_label, loaded_exog


# ------------------------------------------------------------------------------
# 4. ENGENHARIA DE FEATURES ORIENTADA A COMMODITIES
# ------------------------------------------------------------------------------
def build_commodity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói indicadores técnicos, físicos e macroeconômicos para petróleo:
    - Retornos logarítmicos defasados (1d, 5d, 21d, 63d, 126d, 252d)
    - Distância relativa às Médias Móveis Exponenciais (EMA 10, 21, 63, 126, 200)
    - Volatilidade histórica realizada anualizada (21d e 63d)
    - Amplitude diária High-Low normalizada, RSI-14 e Volume Relativo
    - Desvio logarítmico das medianas móveis de 3 e 5 anos (ancoragem cíclica)
    - Exógenas: DXY, Spread Brent-WTI, Crack Spread 3-2-1 e Sazonalidade Cíclica
    """
    feat = pd.DataFrame(index=df.index)
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"].replace(0, np.nan).ffill().fillna(1.0)

    log_ret = np.log(c / c.shift(1))
    feat["Retorno_1d"] = log_ret
    feat["Retorno_5d"] = np.log(c / c.shift(5))
    feat["Retorno_21d"] = np.log(c / c.shift(21))
    feat["Retorno_63d"] = np.log(c / c.shift(63))
    feat["Retorno_126d"] = np.log(c / c.shift(126))
    feat["Retorno_252d"] = np.log(c / c.shift(252))

    for span in [10, 21, 63, 126, 200]:
        ema = c.ewm(span=span, adjust=False).mean()
        feat[f"Dist_EMA_{span}"] = (c - ema) / ema

    feat["Vol_Hist_21d"] = log_ret.rolling(21).std() * np.sqrt(252)
    feat["Vol_Hist_63d"] = log_ret.rolling(63).std() * np.sqrt(252)

    hl_amp = (h - l) / c
    feat["Amplitude_HL"] = hl_amp
    feat["Amplitude_HL_21d"] = hl_amp.rolling(21).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    feat["RSI_14"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0) / 100.0

    vol_ma21 = v.rolling(21).mean()
    feat["Volume_Rel_21d"] = np.log1p(v / vol_ma21.replace(0, np.nan)).fillna(0.0)

    med_3y = c.rolling(252 * 3, min_periods=126).median()
    med_5y = c.rolling(252 * 5, min_periods=252).median()
    feat["Desvio_Mediana_3A"] = np.log(c / med_3y)
    feat["Desvio_Mediana_5A"] = np.log(c / med_5y)

    # Exógenas Macro e Físicas
    dxy = df["DXY"].replace(0, np.nan).ffill().bfill()
    feat["DXY_Ret_21d"] = np.log(dxy / dxy.shift(21)).fillna(0.0)
    feat["DXY_Ret_63d"] = np.log(dxy / dxy.shift(63)).fillna(0.0)
    dxy_ema63 = dxy.ewm(span=63, adjust=False).mean()
    feat["DXY_Dist_EMA63"] = ((dxy - dxy_ema63) / dxy_ema63).fillna(0.0)

    brent_usd = df["Close_USD"].replace(0, np.nan).ffill().bfill()
    wti_usd = df["WTI_USD"].replace(0, np.nan).ffill().bfill()
    spread_bw = (brent_usd - wti_usd) / brent_usd
    feat["Spread_Brent_WTI"] = spread_bw.fillna(0.0)
    feat["Spread_BW_Var21d"] = spread_bw.diff(21).fillna(0.0)

    rb_usd = df["RB_USD"].replace(0, np.nan).ffill().bfill()
    ho_usd = df["HO_USD"].replace(0, np.nan).ffill().bfill()
    crack_321 = ((2.0 * rb_usd * 42.0 + 1.0 * ho_usd * 42.0) / 3.0 - brent_usd) / brent_usd
    feat["Crack_Spread_321"] = crack_321.fillna(0.0)
    feat["Crack_Spread_Var21d"] = crack_321.diff(21).fillna(0.0)

    doy = df.index.dayofyear.values
    feat["Sazonalidade_Sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    feat["Sazonalidade_Cos"] = np.cos(2.0 * np.pi * doy / 365.25)

    return feat.replace([np.inf, -np.inf], np.nan).dropna()


# ------------------------------------------------------------------------------
# 5. MODELO 1: CURTO E MÉDIO PRAZO (2 ANOS) — ENSEMBLE + BANDAS 50%, 80%, 95%
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_tactical_2y_model(
    df: pd.DataFrame,
    marginal_cost_local: float,
    opec_discipline: float,
    geopolitical_risk: float,
    cutoff_date_str: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Modelo Tático Híbrido de 24 Meses (504 dias úteis):
    - Combina Ensemble Regularizado (Ridge + BayesianRidge) sobre features
      técnicas/macro/físicas com atração gravitacional de meio-reversão
      estrutural (custo marginal + mediana de 5 anos), reduzindo o erro
      out-of-sample em horizontes de 6 a 24 meses.
    - Calibra bandas assimétricas de 50% (P25–P75), 80% (P10–P90) e 95%
      (P2.5–P97.5) via Split Conformal Prediction centrado nos resíduos reais.
    """
    if cutoff_date_str is not None:
        df_train = df.loc[: pd.Timestamp(cutoff_date_str)].copy()
    else:
        df_train = df.copy()

    feats = build_commodity_features(df_train)
    close_aligned = df_train.loc[feats.index, "Close"]
    vol_series = feats["Vol_Hist_21d"].clip(lower=0.08)

    horizons = [1, 5, 10, 21, 42, 63, 95, 126, 189, 252, 315, 378, 441, 504]
    X_latest = feats.iloc[[-1]].values
    vol_now = float(vol_series.iloc[-1])
    vol_median = float(vol_series.median())
    vol_regime_mult = float(np.clip((vol_now / max(vol_median, 1e-6)) ** 0.30, 0.85, 1.25))

    current_price = float(close_aligned.iloc[-1])
    last_date = close_aligned.index[-1]

    # Âncora estrutural de meio-reversão (custo marginal + mediana histórica 5A)
    med_5y_now = float(close_aligned.iloc[-252 * 5 :].median())
    eq_anchor = 0.60 * marginal_cost_local + 0.40 * med_5y_now
    ou_gap_log = np.log(eq_anchor / current_price)

    pred_horizons = [0]
    pred_log_ret = [0.0]
    q025_list, q10_list, q25_list = [0.0], [0.0], [0.0]
    q75_list, q90_list, q975_list = [0.0], [0.0], [0.0]
    feature_importances = np.zeros(feats.shape[1])

    for h in horizons:
        y_target = np.log(close_aligned.shift(-h) / close_aligned)
        valid_mask = ~y_target.isna()

        X_h = feats.loc[valid_mask].values
        y_h = y_target.loc[valid_mask].values
        v_h = vol_series.loc[valid_mask].values

        # Winsorização suave (1.5% - 98.5%) para evitar que anomalias como Abril/2020 distorçam os pesos
        y_lo_cap, y_hi_cap = np.percentile(y_h, [1.5, 98.5])
        y_h_clean = np.clip(y_h, y_lo_cap, y_hi_cap)

        scaler = StandardScaler()
        X_h_scaled = scaler.fit_transform(X_h)
        X_lat_scaled = scaler.transform(X_latest)

        alpha = 120.0 + 1.2 * h
        ridge = Ridge(alpha=alpha).fit(X_h_scaled, y_h_clean)
        bayes = BayesianRidge().fit(X_h_scaled, y_h_clean)

        ml_pred = 0.55 * float(ridge.predict(X_lat_scaled)[0]) + 0.45 * float(
            bayes.predict(X_lat_scaled)[0]
        )

        # Fusão com meio-reversão estrutural crescente no horizonte de 2 anos
        t_yr = h / 252.0
        ou_pull = ou_gap_log * (1.0 - np.exp(-0.55 * t_yr))
        w_ou = float(np.clip(0.25 + 0.35 * (t_yr / 2.0), 0.25, 0.60))
        blended_pred = (1.0 - w_ou) * ml_pred + w_ou * ou_pull

        max_abs_ret = 0.32 * np.sqrt(t_yr) + 0.08
        blended_pred = float(np.clip(blended_pred, -max_abs_ret, max_abs_ret))

        # Calibração Conformal Out-of-Sample centrada para bandas de 50%, 80% e 95%
        split_idx = int(len(X_h_scaled) * 0.75)
        if split_idx > 120 and (len(X_h_scaled) - split_idx) > 40:
            val_ridge = Ridge(alpha=alpha).fit(X_h_scaled[:split_idx], y_h_clean[:split_idx])
            oos_res = y_h_clean[split_idx:] - val_ridge.predict(X_h_scaled[split_idx:])
            oos_v_ratio = np.clip((v_h[split_idx:] / max(vol_median, 1e-6)) ** 0.30, 0.80, 1.30)
            norm_res = oos_res / oos_v_ratio
            norm_res_centered = norm_res - np.median(norm_res)
            qs = (
                np.quantile(norm_res_centered, [0.025, 0.10, 0.25, 0.75, 0.90, 0.975])
                * vol_regime_mult
            )
        else:
            ins_res = y_h_clean - ridge.predict(X_h_scaled)
            ins_centered = ins_res - np.median(ins_res)
            qs = np.quantile(ins_centered, [0.025, 0.10, 0.25, 0.75, 0.90, 0.975])

        pred_horizons.append(h)
        pred_log_ret.append(blended_pred)
        q025_list.append(min(float(qs[0]), -0.025 * np.sqrt(t_yr)))
        q10_list.append(min(float(qs[1]), -0.018 * np.sqrt(t_yr)))
        q25_list.append(min(float(qs[2]), -0.010 * np.sqrt(t_yr)))
        q75_list.append(max(float(qs[3]), 0.010 * np.sqrt(t_yr)))
        q90_list.append(max(float(qs[4]), 0.018 * np.sqrt(t_yr)))
        q975_list.append(max(float(qs[5]), 0.025 * np.sqrt(t_yr)))

        if h in [21, 126, 252, 504]:
            feature_importances += np.abs(ridge.coef_) / 4.0

    future_dates = pd.bdate_range(start=last_date, periods=505)
    grid_h = np.arange(0, 505)

    interp_quant = np.interp(grid_h, pred_horizons, pred_log_ret)
    i_025 = -np.maximum.accumulate(-np.interp(grid_h, pred_horizons, q025_list))
    i_10 = -np.maximum.accumulate(-np.interp(grid_h, pred_horizons, q10_list))
    i_25 = -np.maximum.accumulate(-np.interp(grid_h, pred_horizons, q25_list))
    i_75 = np.maximum.accumulate(np.interp(grid_h, pred_horizons, q75_list))
    i_90 = np.maximum.accumulate(np.interp(grid_h, pred_horizons, q90_list))
    i_975 = np.maximum.accumulate(np.interp(grid_h, pred_horizons, q975_list))

    # Choques Macro / Notícias
    t_years = grid_h / 252.0
    opec_curve = 1.0 - np.exp(-t_years / 0.35)
    geo_curve = (1.0 - np.exp(-t_years / 0.12)) * np.exp(-0.22 * np.maximum(0.0, t_years - 0.5))

    opec_log_shift = opec_discipline * 0.16 * opec_curve
    geo_log_shift = (geopolitical_risk / 100.0) * 0.20 * geo_curve
    total_log_ret = interp_quant + opec_log_shift + geo_log_shift

    fc_quant_only = current_price * np.exp(interp_quant)
    fc_central = current_price * np.exp(total_log_ret)

    fc_df = pd.DataFrame(
        {
            "Date": future_dates,
            "Base_Quantitativa_Pura": fc_quant_only,
            "Previsao_Central": fc_central,
            "Banda_Inf_50": current_price * np.exp(total_log_ret + i_25),
            "Banda_Sup_50": current_price * np.exp(total_log_ret + i_75),
            "Banda_Inf_80": current_price * np.exp(total_log_ret + i_10),
            "Banda_Sup_80": current_price * np.exp(total_log_ret + i_90),
            "Banda_Inferior_95": current_price * np.exp(total_log_ret + i_025),
            "Banda_Superior_95": current_price * np.exp(total_log_ret + i_975),
            "Impacto_OPEP_Pct": (np.exp(opec_log_shift) - 1.0) * 100.0,
            "Impacto_Geopolitico_Pct": (np.exp(geo_log_shift) - 1.0) * 100.0,
        }
    ).set_index("Date")

    imp_df = (
        pd.DataFrame(
            {
                "Indicador": feats.columns,
                "Peso_Relativo": (feature_importances / np.sum(feature_importances)) * 100.0,
            }
        )
        .sort_values("Peso_Relativo", ascending=True)
        .reset_index(drop=True)
    )

    meta_info = {
        "current_price": current_price,
        "last_date": last_date,
        "price_12m": float(fc_central[252]),
        "price_24m": float(fc_central[504]),
        "low50_24m": float(fc_df["Banda_Inf_50"].iloc[-1]),
        "high50_24m": float(fc_df["Banda_Sup_50"].iloc[-1]),
        "low80_24m": float(fc_df["Banda_Inf_80"].iloc[-1]),
        "high80_24m": float(fc_df["Banda_Sup_80"].iloc[-1]),
        "lower_24m": float(fc_df["Banda_Inferior_95"].iloc[-1]),
        "upper_24m": float(fc_df["Banda_Superior_95"].iloc[-1]),
        "opec_24m_pct": float((np.exp(opec_log_shift[504]) - 1.0) * 100.0),
        "geo_24m_pct": float((np.exp(geo_log_shift[504]) - 1.0) * 100.0),
        "vol_regime_mult": vol_regime_mult,
    }

    return fc_df, imp_df, meta_info


@st.cache_data(show_spinner=False)
def evaluate_walk_forward_backtest(
    df: pd.DataFrame,
    marginal_cost_local: float,
    cutoff_date_str: str,
    opec_discipline: float = 0.0,
    geopolitical_risk: float = 0.0,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Executa validação Walk-Forward Out-of-Sample treinando o modelo estritamente
    até `cutoff_date_str` e comparando os 2 anos seguintes com os preços reais.
    """
    fc_bt, _, _ = run_tactical_2y_model(
        df=df,
        marginal_cost_local=marginal_cost_local,
        opec_discipline=opec_discipline,
        geopolitical_risk=geopolitical_risk,
        cutoff_date_str=cutoff_date_str,
    )

    cutoff_ts = pd.Timestamp(cutoff_date_str)
    actual_after = df.loc[df.index >= cutoff_ts, ["Close"]].copy()

    comp_df = fc_bt[
        [
            "Previsao_Central",
            "Banda_Inf_50",
            "Banda_Sup_50",
            "Banda_Inf_80",
            "Banda_Sup_80",
            "Banda_Inferior_95",
            "Banda_Superior_95",
        ]
    ].join(actual_after, how="inner")
    comp_df.rename(columns={"Close": "Preco_Real"}, inplace=True)

    eval_slice = comp_df.iloc[1:] if len(comp_df) > 1 else comp_df

    y_true = eval_slice["Preco_Real"].values
    y_pred = eval_slice["Previsao_Central"].values

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)
    accuracy_index = float(np.clip(100.0 - mape, 0.0, 99.9))

    cov_50 = float(
        (
            (eval_slice["Preco_Real"] >= eval_slice["Banda_Inf_50"])
            & (eval_slice["Preco_Real"] <= eval_slice["Banda_Sup_50"])
        ).mean()
        * 100.0
    )
    cov_80 = float(
        (
            (eval_slice["Preco_Real"] >= eval_slice["Banda_Inf_80"])
            & (eval_slice["Preco_Real"] <= eval_slice["Banda_Sup_80"])
        ).mean()
        * 100.0
    )
    cov_95 = float(
        (
            (eval_slice["Preco_Real"] >= eval_slice["Banda_Inferior_95"])
            & (eval_slice["Preco_Real"] <= eval_slice["Banda_Superior_95"])
        ).mean()
        * 100.0
    )

    anchor_price = float(comp_df["Preco_Real"].iloc[0])
    end_price = float(y_true[-1])
    realized_ret_pct = ((end_price / anchor_price) - 1.0) * 100.0

    if realized_ret_pct > 12.0:
        regime_label = "Alta (Bull)"
    elif realized_ret_pct < -12.0:
        regime_label = "Queda (Bear)"
    else:
        regime_label = "Lateral (Range)"

    dir_true = np.sign(y_true - anchor_price)
    dir_pred = np.sign(y_pred - anchor_price)
    dir_acc = float((dir_true == dir_pred).mean() * 100.0)

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "accuracy_index": accuracy_index,
        "cov_50": cov_50,
        "cov_80": cov_80,
        "band_coverage": cov_95,
        "dir_acc": dir_acc,
        "realized_ret_pct": realized_ret_pct,
        "regime": regime_label,
        "n_obs": int(len(eval_slice)),
        "start_eval": eval_slice.index.min(),
        "end_eval": eval_slice.index.max(),
    }
    return comp_df, metrics


@st.cache_data(show_spinner=False)
def run_multi_window_regime_audit(
    df: pd.DataFrame, marginal_cost_local: float, n_windows: int = 8
) -> pd.DataFrame:
    """
    Auditoria Histórica Multi-Janela em 8 ciclos deslizantes de 2 anos.
    """
    min_train_idx = 252 * 4
    max_cutoff_idx = len(df) - 380
    if max_cutoff_idx <= min_train_idx:
        return pd.DataFrame()

    cutoff_indices = np.linspace(min_train_idx, max_cutoff_idx, n_windows, dtype=int)
    rows = []

    for idx in cutoff_indices:
        cutoff_str = df.index[idx].strftime("%Y-%m-%d")
        _, m = evaluate_walk_forward_backtest(
            df=df,
            marginal_cost_local=marginal_cost_local,
            cutoff_date_str=cutoff_str,
            opec_discipline=0.0,
            geopolitical_risk=0.0,
        )
        rows.append(
            {
                "Data_Corte": cutoff_str,
                "Fim_Avaliacao": m["end_eval"].strftime("%Y-%m-%d"),
                "Regime_Mercado": m["regime"],
                "Retorno_Real_Periodo_Pct": m["realized_ret_pct"],
                "RMSE": m["rmse"],
                "MAPE_Pct": m["mape"],
                "Acuracia_Pct": m["accuracy_index"],
                "Cobertura_IC50_Pct": m["cov_50"],
                "Cobertura_IC80_Pct": m["cov_80"],
                "Cobertura_IC95_Pct": m["band_coverage"],
                "Acuracia_Direcional_Pct": m["dir_acc"],
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# 6. MODELO 2: LONGO PRAZO ESTRUTURAL (SCHWARTZ-SMITH OU + JUMP-DIFFUSION)
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_structural_ou_monte_carlo(
    df: pd.DataFrame,
    marginal_cost_usd: float,
    latest_fx_rate: float,
    energy_transition_drift_pct: float = -0.5,
    jump_intensity_mult: float = 1.0,
    horizon_years: int = 10,
    n_sims: int = 5000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Simulação Estrutural de Longo Prazo (Schwartz-Smith OU + Jump-Diffusion):
    - Calibrada sobre retornos semanais para filtrar ruído diário de microestrutura
      e isolar a volatilidade estrutural do equilíbrio de longo prazo, produzindo
      cones de 50% (P25–P75), 80% (P10–P90) e 95% (P2.5–P97.5) realistas e
      ancorados no custo marginal de produção.
    """
    rng = np.random.default_rng(seed)

    weekly_close = df["Close"].resample("W-FRI").last().dropna()
    log_w = np.log(weekly_close.values)
    dt_w = 1.0 / 52.0

    w_rets = np.diff(log_w)
    w_std = float(np.std(w_rets, ddof=1))
    w_mean = float(np.mean(w_rets))

    # Calibração dos Saltos de Poisson em semanas extremas (> 3 sigma)
    jump_mask = np.abs(w_rets - w_mean) > (3.0 * w_std)
    n_jumps = int(np.sum(jump_mask))
    total_years = max(1.0, len(w_rets) / 52.0)

    if n_jumps >= 2:
        lambda_emp = float(np.clip((n_jumps / total_years) * jump_intensity_mult, 0.0, 2.5))
        jump_sizes = w_rets[jump_mask] - w_mean
        mu_j = float(np.clip(np.mean(jump_sizes), -0.10, 0.10))
        sigma_j = float(np.clip(np.std(jump_sizes, ddof=1), 0.03, 0.12))
    else:
        lambda_emp = 0.65 * jump_intensity_mult
        mu_j = -0.015
        sigma_j = 0.065

    # Calibração da difusão estrutural OU nos períodos normais
    clean_mask = ~jump_mask
    x_t = log_w[:-1][clean_mask]
    x_next = log_w[1:][clean_mask]
    b, a = np.polyfit(x_t, x_next, 1)

    # Meia-vida estrutural realista entre ~0.9 e ~1.65 anos (kappa in [0.42, 0.78])
    b_min = np.exp(-0.78 * dt_w)
    b_max = np.exp(-0.42 * dt_w)
    b = float(np.clip(b, b_min, b_max))
    kappa = -np.log(b) / dt_w

    resid = x_next - (a + b * x_t)
    sigma_eps = float(np.std(resid, ddof=2))
    raw_sigma = sigma_eps * np.sqrt(2.0 * kappa / (1.0 - b**2))
    # Fator estrutural de equilíbrio de longo prazo (Schwartz-Smith 2000)
    sigma = float(np.clip(raw_sigma * 0.68, 0.14, 0.24))

    empirical_theta = a / (1.0 - b)
    marginal_cost_local = marginal_cost_usd * latest_fx_rate
    theta_0 = 0.70 * np.log(marginal_cost_local) + 0.30 * empirical_theta

    current_price = float(df["Close"].iloc[-1])
    x0 = np.log(current_price)
    last_date = df.index[-1]

    steps_per_year = 12
    n_steps = horizon_years * steps_per_year
    dt = 1.0 / steps_per_year

    exp_kdt = np.exp(-kappa * dt)
    one_minus_exp = 1.0 - exp_kdt
    step_std = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))
    jump_drift_comp = lambda_emp * mu_j * (one_minus_exp / max(kappa, 1e-6))

    X = np.zeros((n_sims, n_steps + 1), dtype=np.float64)
    X[:, 0] = x0

    z_shocks = rng.standard_normal((n_sims, n_steps))
    n_poisson = rng.poisson(lambda_emp * dt, size=(n_sims, n_steps))
    z_jump = rng.standard_normal((n_sims, n_steps))
    jump_component = n_poisson * mu_j + np.sqrt(n_poisson) * sigma_j * z_jump

    annual_drift = energy_transition_drift_pct / 100.0
    for step in range(1, n_steps + 1):
        t_yr = step * dt
        theta_t = theta_0 + annual_drift * t_yr
        X[:, step] = (
            X[:, step - 1] * exp_kdt
            + theta_t * one_minus_exp
            - jump_drift_comp
            + step_std * z_shocks[:, step - 1]
            + jump_component[:, step - 1]
        )

    S = np.exp(X)
    future_dates = pd.DatetimeIndex(
        [last_date + pd.DateOffset(months=step) for step in range(n_steps + 1)]
    )

    p025, p10, p25, p50, p75, p90, p975 = np.percentile(
        S, [2.5, 10, 25, 50, 75, 90, 97.5], axis=0
    )

    mc_df = pd.DataFrame(
        {
            "Date": future_dates,
            "Ano": np.arange(0, n_steps + 1) / steps_per_year,
            "Percentil_025": p025,
            "Cenario_Pessimista_P10": p10,
            "Percentil_25": p25,
            "Cenario_Base_P50": p50,
            "Percentil_75": p75,
            "Cenario_Otimista_P90": p90,
            "Percentil_975": p975,
            "Media_Monte_Carlo": np.mean(S, axis=0),
        }
    ).set_index("Date")

    total_var_rate = sigma**2 + lambda_emp * (mu_j**2 + sigma_j**2)
    stationary_std_log = np.sqrt(total_var_rate / (2.0 * kappa))
    z_score = float((x0 - theta_0) / stationary_std_log)

    ou_stats = {
        "kappa": float(kappa),
        "half_life_years": float(np.log(2.0) / kappa),
        "sigma_annual": float(sigma),
        "lambda_jumps": float(lambda_emp),
        "mu_jump_pct": float((np.exp(mu_j) - 1.0) * 100.0),
        "sigma_jump_pct": float(sigma_j * 100.0),
        "equilibrium_price_0": float(np.exp(theta_0)),
        "stationary_std_log": float(stationary_std_log),
        "upper_2sigma_eq": float(np.exp(theta_0 + 2.0 * stationary_std_log)),
        "lower_2sigma_eq": float(np.exp(theta_0 - 2.0 * stationary_std_log)),
        "z_score": z_score,
        "terminal_p975": float(p975[-1]),
        "terminal_p90": float(p90[-1]),
        "terminal_p75": float(p75[-1]),
        "terminal_p50": float(p50[-1]),
        "terminal_p25": float(p25[-1]),
        "terminal_p10": float(p10[-1]),
        "terminal_p025": float(p025[-1]),
        "sample_paths": S[:25, :],
        "sim_matrix": S,
    }

    return mc_df, ou_stats


# ------------------------------------------------------------------------------
# 7. FUNÇÕES DE VISUALIZAÇÃO INTERATIVA (LAYOUT SEM SOBREPOSIÇÃO)
# ------------------------------------------------------------------------------
def _apply_institutional_layout(
    fig: go.Figure,
    title_text: str,
    height: int = 520,
    yaxis_title: str = "",
    xaxis_title: str = "",
) -> go.Figure:
    """
    Aplica layout padronizado com âncora de título no topo do container
    e margem superior ampla (t=118), impedindo qualquer sobreposição entre
    o título e legendas horizontais de múltiplas linhas.
    """
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.01,
            xanchor="left",
            y=0.98,
            yref="container",
            yanchor="top",
            font=dict(size=16, color="#F8FAFC"),
        ),
        template="plotly_dark",
        height=height,
        hovermode="x unified",
        yaxis_title=yaxis_title,
        xaxis_title=xaxis_title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(15, 23, 42, 0.55)",
            font=dict(size=11),
        ),
        margin=dict(l=15, r=15, t=118, b=28),
    )
    return fig


def plot_zscore_gauge(z_score: float) -> go.Figure:
    """Cria velocímetro institucional do Z-Score estatístico do Brent."""
    clamped_z = float(np.clip(z_score, -3.5, 3.5))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=z_score,
            number={"valueformat": "+.2f", "font": {"size": 34, "color": "#F8FAFC"}},
            title={
                "text": "Z-Score Estrutural vs. Equilíbrio",
                "font": {"size": 14, "color": "#CBD5E1"},
            },
            gauge={
                "axis": {"range": [-3.5, 3.5], "tickwidth": 1, "tickcolor": "#94A3B8"},
                "bar": {"color": "#F8FAFC", "thickness": 0.25},
                "bgcolor": "#0F172A",
                "borderwidth": 1,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [-3.5, -2.0], "color": "rgba(56, 189, 248, 0.45)"},
                    {"range": [-2.0, -1.0], "color": "rgba(16, 185, 129, 0.22)"},
                    {"range": [-1.0, 1.0], "color": "rgba(16, 185, 129, 0.40)"},
                    {"range": [1.0, 2.0], "color": "rgba(245, 158, 11, 0.28)"},
                    {"range": [2.0, 3.5], "color": "rgba(239, 68, 68, 0.48)"},
                ],
                "threshold": {
                    "line": {"color": "#FFFFFF", "width": 3},
                    "thickness": 0.8,
                    "value": clamped_z,
                },
            },
        )
    )
    fig.update_layout(
        height=265,
        margin=dict(l=20, r=20, t=55, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
    )
    return fig


def plot_historical_ohlcv(
    df: pd.DataFrame, currency_code: str, ou_stats: Dict, lookback_label: str
) -> go.Figure:
    """Gráfico Candlestick OHLCV com EMAs e Bandas de Equilíbrio Z=±2."""
    window_map = {
        "1 Ano": 252,
        "3 Anos": 252 * 3,
        "5 Anos": 252 * 5,
        "10 Anos": 252 * 10,
        "Histórico Completo": len(df),
    }
    n_rows = window_map.get(lookback_label, 252 * 3)
    sub = df.iloc[-n_rows:].copy()
    sub["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean().loc[sub.index]
    sub["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean().loc[sub.index]

    sym = CURRENCIES[currency_code]["symbol"]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.76, 0.24],
    )

    fig.add_trace(
        go.Candlestick(
            x=sub.index,
            open=sub["Open"],
            high=sub["High"],
            low=sub["Low"],
            close=sub["Close"],
            name=f"Brent OHLC ({currency_code})",
            increasing_line_color="#10B981",
            decreasing_line_color="#EF4444",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=sub.index,
            y=sub["EMA_21"],
            mode="lines",
            name="EMA 21d",
            line=dict(color="#38BDF8", width=1.4),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=sub.index,
            y=sub["EMA_200"],
            mode="lines",
            name="EMA 200d",
            line=dict(color="#F59E0B", width=1.8),
        ),
        row=1,
        col=1,
    )

    fig.add_hline(
        y=ou_stats["equilibrium_price_0"],
        line_dash="dash",
        line_color="#A855F7",
        annotation_text=f"Equilíbrio (Z=0): {format_price(ou_stats['equilibrium_price_0'], currency_code)}",
        annotation_position="top left",
        row=1,
        col=1,
    )
    fig.add_hline(
        y=ou_stats["upper_2sigma_eq"],
        line_dash="dot",
        line_color="#EF4444",
        annotation_text=f"Sobrecomprado (Z=+2): {format_price(ou_stats['upper_2sigma_eq'], currency_code)}",
        annotation_position="top right",
        row=1,
        col=1,
    )
    fig.add_hline(
        y=ou_stats["lower_2sigma_eq"],
        line_dash="dot",
        line_color="#38BDF8",
        annotation_text=f"Subavaliado (Z=-2): {format_price(ou_stats['lower_2sigma_eq'], currency_code)}",
        annotation_position="bottom right",
        row=1,
        col=1,
    )

    colors_vol = np.where(
        sub["Close"] >= sub["Open"], "rgba(16,185,129,0.5)", "rgba(239,68,68,0.5)"
    )
    fig.add_trace(
        go.Bar(
            x=sub.index,
            y=sub["Volume"],
            name="Volume (Contratos)",
            marker_color=colors_vol,
        ),
        row=2,
        col=1,
    )

    _apply_institutional_layout(
        fig,
        title_text=f"Série Histórica Brent Crude (BZ=F) em {currency_code} — Câmbio Diário Exato",
        height=560,
    )
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text=f"Preço ({sym}/bbl)", row=1, col=1)
    fig.update_yaxes(title_text="Contratos", row=2, col=1)
    return fig


def plot_tactical_2y_forecast(
    df: pd.DataFrame,
    fc_df: pd.DataFrame,
    currency_code: str,
    show_band_95: bool = True,
    history_days: int = 504,
) -> go.Figure:
    """
    Gráfico 1: Histórico recente + Projeção Tática de 2 Anos com bandas
    aninhadas de 50% (P25-P75), 80% (P10-P90) e 95% (P2.5-P97.5).
    """
    hist_slice = df.iloc[-history_days:]
    sym = CURRENCIES[currency_code]["symbol"]

    fig = go.Figure()

    # Histórico realizado
    fig.add_trace(
        go.Scatter(
            x=hist_slice.index,
            y=hist_slice["Close"],
            mode="lines",
            name="Histórico Realizado",
            line=dict(color="#E2E8F0", width=2.0),
        )
    )

    # Banda Externa 95% (Opcional)
    if show_band_95:
        fig.add_trace(
            go.Scatter(
                x=fc_df.index,
                y=fc_df["Banda_Superior_95"],
                mode="lines",
                line=dict(color="rgba(56, 189, 248, 0.25)", width=1, dash="dot"),
                showlegend=False,
                name="Sup. 95%",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc_df.index,
                y=fc_df["Banda_Inferior_95"],
                mode="lines",
                name="Banda 95% (P2.5–P97.5)",
                line=dict(color="rgba(56, 189, 248, 0.25)", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(56, 189, 248, 0.08)",
            )
        )

    # Banda Intermediária 80% (P10 - P90)
    fig.add_trace(
        go.Scatter(
            x=fc_df.index,
            y=fc_df["Banda_Sup_80"],
            mode="lines",
            line=dict(color="rgba(56, 189, 248, 0.45)", width=1, dash="dash"),
            showlegend=False,
            name="Sup. 80%",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_df.index,
            y=fc_df["Banda_Inf_80"],
            mode="lines",
            name="Banda 80% (P10–P90)",
            line=dict(color="rgba(56, 189, 248, 0.45)", width=1, dash="dash"),
            fill="tonexty",
            fillcolor="rgba(56, 189, 248, 0.15)",
        )
    )

    # Banda Interna 50% (P25 - P75 — Núcleo de Alta Probabilidade)
    fig.add_trace(
        go.Scatter(
            x=fc_df.index,
            y=fc_df["Banda_Sup_50"],
            mode="lines",
            line=dict(color="rgba(16, 185, 129, 0.55)", width=1),
            showlegend=False,
            name="Sup. 50%",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_df.index,
            y=fc_df["Banda_Inf_50"],
            mode="lines",
            name="Banda 50% (P25–P75)",
            line=dict(color="rgba(16, 185, 129, 0.55)", width=1),
            fill="tonexty",
            fillcolor="rgba(16, 185, 129, 0.24)",
        )
    )

    # Linha Base Quantitativa sem Choques Macro
    fig.add_trace(
        go.Scatter(
            x=fc_df.index,
            y=fc_df["Base_Quantitativa_Pura"],
            mode="lines",
            name="Base Quantitativa (Sem Choques)",
            line=dict(color="#94A3B8", width=1.4, dash="dash"),
        )
    )

    # Linha Central Ajustada
    fig.add_trace(
        go.Scatter(
            x=fc_df.index,
            y=fc_df["Previsao_Central"],
            mode="lines",
            name="Projeção Central (Mediana)",
            line=dict(color="#38BDF8", width=3.0),
        )
    )

    fig.add_vline(x=hist_slice.index[-1], line_dash="dash", line_color="#F59E0B")

    return _apply_institutional_layout(
        fig,
        title_text=f"Modelo 1: Projeção Tática de 2 Anos ({currency_code}) — Cones de 50%, 80% e 95%",
        height=540,
        yaxis_title=f"Preço do Barril ({sym})",
        xaxis_title="Data",
    )


def plot_walk_forward_backtest(
    df: pd.DataFrame, comp_df: pd.DataFrame, currency_code: str
) -> go.Figure:
    """Gráfico de Validação Out-of-Sample Walk-Forward com bandas de 50% e 80%."""
    cutoff_date = comp_df.index[0]
    pre_context = df.loc[:cutoff_date].iloc[-252:]
    sym = CURRENCIES[currency_code]["symbol"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pre_context.index,
            y=pre_context["Close"],
            mode="lines",
            name="Treino (Contexto Pré-Corte)",
            line=dict(color="#64748B", width=1.6),
        )
    )

    # Banda 80%
    fig.add_trace(
        go.Scatter(
            x=comp_df.index,
            y=comp_df["Banda_Sup_80"],
            mode="lines",
            line=dict(color="rgba(168, 85, 247, 0.30)", width=1, dash="dot"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=comp_df.index,
            y=comp_df["Banda_Inf_80"],
            mode="lines",
            name="Banda 80% Out-of-Sample",
            line=dict(color="rgba(168, 85, 247, 0.30)", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(168, 85, 247, 0.12)",
        )
    )

    # Banda 50%
    fig.add_trace(
        go.Scatter(
            x=comp_df.index,
            y=comp_df["Banda_Sup_50"],
            mode="lines",
            line=dict(color="rgba(168, 85, 247, 0.50)", width=1),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=comp_df.index,
            y=comp_df["Banda_Inf_50"],
            mode="lines",
            name="Banda 50% Out-of-Sample",
            line=dict(color="rgba(168, 85, 247, 0.50)", width=1),
            fill="tonexty",
            fillcolor="rgba(168, 85, 247, 0.22)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=comp_df.index,
            y=comp_df["Preco_Real"],
            mode="lines",
            name="Preço Realizado no Mercado",
            line=dict(color="#10B981", width=2.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=comp_df.index,
            y=comp_df["Previsao_Central"],
            mode="lines",
            name="Previsão Out-of-Sample (2 Anos)",
            line=dict(color="#F59E0B", width=2.5, dash="dash"),
        )
    )

    fig.add_vline(x=cutoff_date, line_dash="dash", line_color="#EF4444")

    return _apply_institutional_layout(
        fig,
        title_text=f"Backtesting Walk-Forward: Previsão vs. Realizado ({currency_code})",
        height=490,
        yaxis_title=f"Preço ({sym}/bbl)",
    )


def plot_structural_monte_carlo_10y(
    df: pd.DataFrame,
    mc_df: pd.DataFrame,
    ou_stats: Dict,
    currency_code: str,
    show_band_95: bool = True,
    show_sample_paths: bool = False,
) -> go.Figure:
    """
    Gráfico 2: Cone de Probabilidade Estrutural de Longo Prazo (5.000 simulações)
    exibindo bandas de 50% (P25-P75), 80% (P10-P90) e 95% (P2.5-P97.5).
    """
    hist_recent = df.iloc[-504:]
    sym = CURRENCIES[currency_code]["symbol"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist_recent.index,
            y=hist_recent["Close"],
            mode="lines",
            name="Histórico Recente",
            line=dict(color="#CBD5E1", width=2.0),
        )
    )

    if show_sample_paths and "sample_paths" in ou_stats:
        sample_paths = ou_stats["sample_paths"]
        for i in range(min(12, sample_paths.shape[0])):
            fig.add_trace(
                go.Scatter(
                    x=mc_df.index,
                    y=sample_paths[i, :],
                    mode="lines",
                    line=dict(color="rgba(148, 163, 184, 0.14)", width=0.8),
                    showlegend=(i == 0),
                    name="Trajetórias Estocásticas" if i == 0 else None,
                    hoverinfo="skip",
                )
            )

    # Banda Externa 95% (P2.5 - P97.5)
    if show_band_95:
        fig.add_trace(
            go.Scatter(
                x=mc_df.index,
                y=mc_df["Percentil_975"],
                mode="lines",
                line=dict(color="rgba(148, 163, 184, 0.28)", width=1, dash="dot"),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=mc_df.index,
                y=mc_df["Percentil_025"],
                mode="lines",
                name="Banda 95% (P2.5–P97.5)",
                line=dict(color="rgba(148, 163, 184, 0.28)", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(148, 163, 184, 0.08)",
            )
        )

    # Cenário Otimista P90 -> Mediana P50 (Metade Superior da Banda de 80%)
    fig.add_trace(
        go.Scatter(
            x=mc_df.index,
            y=mc_df["Cenario_Otimista_P90"],
            mode="lines",
            name="Otimista (P90 — Topo 80%)",
            line=dict(color="#10B981", width=2.0, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mc_df.index,
            y=mc_df["Cenario_Base_P50"],
            mode="lines",
            showlegend=False,
            line=dict(color="rgba(245, 158, 11, 0.0)", width=0),
            fill="tonexty",
            fillcolor="rgba(16, 185, 129, 0.12)",
        )
    )

    # Mediana P50 -> Cenário Pessimista P10 (Metade Inferior da Banda de 80%)
    fig.add_trace(
        go.Scatter(
            x=mc_df.index,
            y=mc_df["Cenario_Pessimista_P10"],
            mode="lines",
            name="Pessimista (P10 — Base 80%)",
            line=dict(color="#EF4444", width=2.0, dash="dash"),
            fill="tonexty",
            fillcolor="rgba(239, 68, 68, 0.12)",
        )
    )

    # Banda Interna de 50% (P25 - P75 — Núcleo Estrutural de Alta Probabilidade)
    fig.add_trace(
        go.Scatter(
            x=mc_df.index,
            y=mc_df["Percentil_75"],
            mode="lines",
            line=dict(color="rgba(245, 158, 11, 0.55)", width=1),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mc_df.index,
            y=mc_df["Percentil_25"],
            mode="lines",
            name="Banda 50% (P25–P75)",
            line=dict(color="rgba(245, 158, 11, 0.55)", width=1),
            fill="tonexty",
            fillcolor="rgba(245, 158, 11, 0.24)",
        )
    )

    # Linha Central Mediana (P50)
    fig.add_trace(
        go.Scatter(
            x=mc_df.index,
            y=mc_df["Cenario_Base_P50"],
            mode="lines",
            name="Cenário Base / Mediana (P50)",
            line=dict(color="#F59E0B", width=3.2),
        )
    )

    fig.add_hline(
        y=ou_stats["equilibrium_price_0"],
        line_dash="dot",
        line_color="#A855F7",
        annotation_text=f"Custo Marginal / Equilíbrio Inicial: {format_price(ou_stats['equilibrium_price_0'], currency_code)}",
        annotation_position="bottom left",
    )

    return _apply_institutional_layout(
        fig,
        title_text=f"Modelo 2: Cone Estrutural OU + Jump-Diffusion (5.000 Simulações — {currency_code})",
        height=550,
        yaxis_title=f"Preço Projetado ({sym}/bbl)",
        xaxis_title="Horizonte Estrutural",
    )


def plot_hedge_distribution(
    unhedged_cashflows: np.ndarray,
    hedged_cashflows: np.ndarray,
    currency_code: str,
    profile_label: str,
) -> go.Figure:
    """Histograma comparativo de distribuição financeira Sem Hedge vs. Com Hedge."""
    sym = CURRENCIES[currency_code]["symbol"]
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=unhedged_cashflows / 1e6,
            name="Sem Hedge (Exposição Pura)",
            marker_color="rgba(239, 68, 68, 0.55)",
            nbinsx=45,
        )
    )

    fig.add_trace(
        go.Histogram(
            x=hedged_cashflows / 1e6,
            name="Com Hedge Estruturado",
            marker_color="rgba(16, 185, 129, 0.65)",
            nbinsx=45,
        )
    )

    _apply_institutional_layout(
        fig,
        title_text=f"Distribuição de Impacto Financeiro Acumulado ({profile_label}) — Milhões de {currency_code}",
        height=430,
        xaxis_title=f"Fluxo Financeiro Total ({sym} Milhões)",
        yaxis_title="Frequência (Simulações Monte Carlo)",
    )
    fig.update_layout(barmode="overlay")
    return fig


# ------------------------------------------------------------------------------
# 8. INTERFACE PRINCIPAL E CONTROLES DA BARRA LATERAL
# ------------------------------------------------------------------------------
def main():
    st.sidebar.markdown("## ⚙️ Parâmetros Quantitativos")

    currency_options = list(CURRENCIES.keys())
    selected_currency = st.sidebar.selectbox(
        "💱 Moeda de Cotação e Análise",
        options=currency_options,
        index=0,
        format_func=lambda c: CURRENCIES[c]["name"],
        help="Converte cada ponto histórico diário do Brent pela taxa de câmbio exata daquele respectivo dia.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Modelo 1: Choques Táticos (2 Anos)")
    opec_discipline = st.sidebar.slider(
        "Fator de Disciplina da OPEP+",
        min_value=-1.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="-1.0 = Guerra de Preços / Excesso de Oferta | 0.0 = Neutro | +1.0 = Cortes Rigorosos de Produção",
    )
    geopolitical_risk = st.sidebar.slider(
        "Índice de Risco Geopolítico / Notícias",
        min_value=-100,
        max_value=100,
        value=0,
        step=5,
        help="-100 = Distensão / Acordos de Paz | 0 = Baseline Histórico | +100 = Choque Severo de Oferta / Conflito",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌐 Modelo 2: Estrutural (10 Anos)")
    horizon_years = st.sidebar.slider(
        "Horizonte de Longo Prazo (Anos)",
        min_value=3,
        max_value=10,
        value=10,
        step=1,
    )
    marginal_cost_usd = st.sidebar.slider(
        "Custo Marginal Global de Produção (USD/bbl)",
        min_value=45.0,
        max_value=110.0,
        value=75.0,
        step=2.5,
        help="Âncora fundamental de equilíbrio (shale oil americano e águas profundas).",
    )
    energy_transition_drift = st.sidebar.slider(
        "Deriva Secular Transição Energética (% a.a.)",
        min_value=-3.0,
        max_value=2.0,
        value=-0.5,
        step=0.25,
        help="Ajuste estrutural anual na curva de equilíbrio (negativo = substituição por renováveis/EVs; positivo = subinvestimento em Capex).",
    )
    jump_intensity_mult = st.sidebar.slider(
        "Multiplicador de Saltos de Poisson (Choques >3σ)",
        min_value=0.0,
        max_value=2.5,
        value=1.0,
        step=0.25,
        help="0.0 = OU Puro sem saltos | 1.0 = Frequência histórica calibrada | >1.0 = Regime de maior ocorrência de cisnes negros.",
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Atualizar Dados de Mercado"):
        st.cache_data.clear()
        st.rerun()

    # Ingestão desacoplada (BZ=F em cache separado da conversão FX)
    try:
        with st.spinner(
            f"Sincronizando Brent (BZ=F) e câmbio diário exato {selected_currency}..."
        ):
            df_brent, fx_label, loaded_exog = get_market_data_in_currency(selected_currency)
    except Exception as err:
        st.error(f"❌ Erro ao carregar dados financeiros: {err}")
        st.info(
            "Verifique sua conexão com a internet e clique em 'Atualizar Dados de Mercado' na barra lateral."
        )
        st.stop()

    latest_fx = float(df_brent["FX_Rate"].iloc[-1])
    current_price = float(df_brent["Close"].iloc[-1])
    prev_price = float(df_brent["Close"].iloc[-2])
    daily_chg_pct = ((current_price / prev_price) - 1.0) * 100.0
    vol_21d = float(
        np.log(df_brent["Close"] / df_brent["Close"].shift(1)).iloc[-21:].std()
        * np.sqrt(252)
        * 100.0
    )
    marginal_cost_local = marginal_cost_usd * latest_fx

    fc_2y_df, imp_2y_df, meta_2y = run_tactical_2y_model(
        df=df_brent,
        marginal_cost_local=marginal_cost_local,
        opec_discipline=opec_discipline,
        geopolitical_risk=geopolitical_risk,
    )

    mc_10y_df, ou_stats = run_structural_ou_monte_carlo(
        df=df_brent,
        marginal_cost_usd=marginal_cost_usd,
        latest_fx_rate=latest_fx,
        energy_transition_drift_pct=energy_transition_drift,
        jump_intensity_mult=jump_intensity_mult,
        horizon_years=horizon_years,
        n_sims=5000,
    )

    default_bt_cutoff = df_brent.index[-504].strftime("%Y-%m-%d")
    _, default_bt_metrics = evaluate_walk_forward_backtest(
        df=df_brent,
        marginal_cost_local=marginal_cost_local,
        cutoff_date_str=default_bt_cutoff,
        opec_discipline=0.0,
        geopolitical_risk=0.0,
    )

    # Cabeçalho Institucional
    st.markdown(
        '<div class="main-header">🛢️ Brent Crude Quant Terminal — Econometria & Previsão</div>',
        unsafe_allow_html=True,
    )
    exog_str = ", ".join(loaded_exog) if loaded_exog else "Séries Técnicas + Sazonalidade"
    st.markdown(
        f'<div class="sub-header">Ativo: <b>ICE Brent Crude (BZ=F)</b> | '
        f'Amostra: <b>{df_brent.index.min().strftime("%d/%m/%Y")}</b> até <b>{df_brent.index.max().strftime("%d/%m/%Y")}</b> '
        f'({len(df_brent):,} pregões) | Moeda: <b>{CURRENCIES[selected_currency]["name"]}</b> [{fx_label}]<br/>'
        f'Fatores Exógenos Ativos: <b>{exog_str} + Sazonalidade Cíclica (Fourier)</b></div>',
        unsafe_allow_html=True,
    )

    # Módulo de Alerta Z-Score em Destaque Global
    z_val = ou_stats["z_score"]
    if z_val > 2.0:
        z_status = "SOBRECOMPRADO / RISCO DE ALTA EXTREMA (Z > 2)"
        z_desc = (
            f"O preço atual ({format_price(current_price, selected_currency)}) está {z_val:+.2f} desvios-padrão "
            f"acima do equilíbrio estrutural de longo prazo ({format_price(ou_stats['equilibrium_price_0'], selected_currency)}). "
            "Probabilidade elevada de compressão de margem e reversão à média."
        )
        banner_class = "zscore-banner-over"
        z_icon = "🚨"
    elif z_val < -2.0:
        z_status = "SUBAVALIADO / OPORTUNIDADE ESTRUTURAL (Z < -2)"
        z_desc = (
            f"O preço atual ({format_price(current_price, selected_currency)}) está {z_val:+.2f} desvios-padrão "
            f"abaixo do custo marginal de equilíbrio ({format_price(ou_stats['equilibrium_price_0'], selected_currency)}). "
            "Forte assimetria altista pela destruição de oferta marginal."
        )
        banner_class = "zscore-banner-under"
        z_icon = "💎"
    else:
        z_status = "JUSTO / EQUILÍBRIO ESTATÍSTICO (-2 ≤ Z ≤ 2)"
        z_desc = (
            f"O preço atual ({format_price(current_price, selected_currency)}) opera dentro da faixa normal de "
            f"±2σ em torno do equilíbrio estrutural ({format_price(ou_stats['equilibrium_price_0'], selected_currency)}), "
            f"com Z-Score de {z_val:+.2f}."
        )
        banner_class = "zscore-banner-fair"
        z_icon = "✅"

    st.markdown(
        f'<div class="{banner_class}">'
        f'<b>{z_icon} DIAGNÓSTICO Z-SCORE DE COMMODITIES: {z_status}</b><br/>'
        f'<span style="font-size:0.90rem; color:#E2E8F0;">{z_desc}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # KPIs Principais
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        f"Brent Spot ({selected_currency})",
        format_price(current_price, selected_currency),
        f"{daily_chg_pct:+.2f}% (1d)",
    )
    k2.metric(
        "Câmbio Diário Atual",
        f"{latest_fx:,.4f} {selected_currency}/USD",
        f"USD Base: ${df_brent['Close_USD'].iloc[-1]:,.2f}",
    )
    k3.metric(
        "Volatilidade Realizada (21d)",
        f"{vol_21d:.1f}% a.a.",
        f"Meia-Vida OU: {ou_stats['half_life_years']:.1f} anos",
    )
    k4.metric(
        "Projeção Tática (2 Anos)",
        format_price(meta_2y["price_24m"], selected_currency),
        f"IC 50%: [{format_price(meta_2y['low50_24m'], selected_currency)} ; {format_price(meta_2y['high50_24m'], selected_currency)}]",
    )
    k5.metric(
        f"Estrutural Base ({horizon_years} Anos)",
        format_price(ou_stats["terminal_p50"], selected_currency),
        f"Z-Score: {z_val:+.2f}",
    )

    # --------------------------------------------------------------------------
    # ORGANIZAÇÃO EM ABAS (st.tabs)
    # --------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Visão Geral & Z-Score",
            "🎯 Modelo 1: Tático (2 Anos)",
            "🔁 Backtesting & Auditoria",
            "🌐 Modelo 2: Estrutural (10 Anos)",
            "🛡️ Risco & Hedge Corporativo",
            "💾 Dados & Exportação CSV",
        ]
    )

    # ==========================================================================
    # ABA 1: VISÃO GERAL, OHLCV E Z-SCORE
    # ==========================================================================
    with tab1:
        col_chart, col_gauge = st.columns([2.3, 1.0])

        with col_chart:
            lookback_choice = st.radio(
                "Janela Histórica OHLCV:",
                options=["1 Ano", "3 Anos", "5 Anos", "10 Anos", "Histórico Completo"],
                index=1,
                horizontal=True,
            )
            fig_ohlcv = plot_historical_ohlcv(
                df=df_brent,
                currency_code=selected_currency,
                ou_stats=ou_stats,
                lookback_label=lookback_choice,
            )
            st.plotly_chart(fig_ohlcv, use_container_width=True)

        with col_gauge:
            st.plotly_chart(plot_zscore_gauge(z_val), use_container_width=True)

            st.markdown("#### 📐 Parâmetros da Distribuição Estacionária")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Métrica": [
                            "Preço Atual",
                            "Equilíbrio Longo Prazo (Z = 0)",
                            "Limite Superior (Z = +2)",
                            "Limite Inferior (Z = -2)",
                            "Velocidade Reversão (κ)",
                            "Vol. Estrutural Difusão (σ)",
                            "Intensidade Saltos (λ)",
                            "Média / Vol. dos Saltos",
                        ],
                        "Valor": [
                            format_price(current_price, selected_currency),
                            format_price(ou_stats["equilibrium_price_0"], selected_currency),
                            format_price(ou_stats["upper_2sigma_eq"], selected_currency),
                            format_price(ou_stats["lower_2sigma_eq"], selected_currency),
                            f"{ou_stats['kappa']:.3f} / ano",
                            f"{ou_stats['sigma_annual'] * 100:.1f}% a.a.",
                            f"{ou_stats['lambda_jumps']:.2f} saltos/ano",
                            f"{ou_stats['mu_jump_pct']:+.1f}% / ±{ou_stats['sigma_jump_pct']:.1f}%",
                        ],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

            try:
                adf_price_p = compute_adf_pvalue(df_brent["Close"].iloc[-1500:])
                adf_ret_p = compute_adf_pvalue(
                    np.log(df_brent["Close"] / df_brent["Close"].shift(1)).dropna().iloc[-1500:]
                )
                st.caption(
                    f"**Teste Dickey-Fuller Aumentado (ADF):** p-valor Nível = `{adf_price_p:.3f}` | "
                    f"p-valor Log-Retornos = `{adf_ret_p:.4f}` (Estacionário)."
                )
            except Exception:
                pass

    # ==========================================================================
    # ABA 2: MODELO 1 — CURTO E MÉDIO PRAZO (2 ANOS)
    # ==========================================================================
    with tab2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Alvo Central (24 Meses)",
            format_price(meta_2y["price_24m"], selected_currency),
            f"12M: {format_price(meta_2y['price_12m'], selected_currency)}",
        )
        m2.metric(
            "Banda 50% (P25–P75) em 24M",
            f"{format_price(meta_2y['low50_24m'], selected_currency)} – {format_price(meta_2y['high50_24m'], selected_currency)}",
            "Núcleo de Alta Probabilidade",
        )
        m3.metric(
            "Banda 80% (P10–P90) em 24M",
            f"{format_price(meta_2y['low80_24m'], selected_currency)} – {format_price(meta_2y['high80_24m'], selected_currency)}",
            f"95%: [{format_price(meta_2y['lower_24m'], selected_currency)} ; {format_price(meta_2y['upper_24m'], selected_currency)}]",
        )
        m4.metric(
            "Índice de Confiança / Acurácia",
            f"{default_bt_metrics['accuracy_index']:.1f}%",
            f"MAPE OOS: {default_bt_metrics['mape']:.2f}% | Choque: {meta_2y['opec_24m_pct'] + meta_2y['geo_24m_pct']:+.1f}%",
        )

        show_2y_95 = st.checkbox(
            "Exibir faixa externa de 95% (P2.5–P97.5) além das bandas de 50% e 80%",
            value=True,
            key="chk_2y_95",
        )

        fig_2y = plot_tactical_2y_forecast(
            df=df_brent,
            fc_df=fc_2y_df,
            currency_code=selected_currency,
            show_band_95=show_2y_95,
            history_days=504,
        )
        st.plotly_chart(fig_2y, use_container_width=True)

        col_imp, col_table2y = st.columns([1.05, 1.35])
        with col_imp:
            top_imp = imp_2y_df.tail(15)
            fig_imp = go.Figure(
                go.Bar(
                    x=top_imp["Peso_Relativo"],
                    y=top_imp["Indicador"],
                    orientation="h",
                    marker_color="#38BDF8",
                )
            )
            _apply_institutional_layout(
                fig_imp,
                title_text="Top 15 Features Mais Relevantes (Técnicas, Macro & Sazonalidade)",
                height=420,
                xaxis_title="Peso Normalizado (%)",
            )
            fig_imp.update_layout(margin=dict(l=15, r=15, t=65, b=25), showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)

        with col_table2y:
            st.markdown("#### 📅 Marcos Trimestrais da Projeção Tática (Bandas 50%, 80% e 95%)")
            step_indices = [0, 63, 126, 189, 252, 315, 378, 441, 504]
            milestones = fc_2y_df.iloc[step_indices].copy()
            milestones["Horizonte"] = [
                "Hoje (t=0)",
                "3 Meses",
                "6 Meses",
                "9 Meses",
                "12 Meses (1A)",
                "15 Meses",
                "18 Meses",
                "21 Meses",
                "24 Meses (2A)",
            ]
            disp_2y = pd.DataFrame(
                {
                    "Horizonte": milestones["Horizonte"].values,
                    "Data": milestones.index.strftime("%Y-%m-%d"),
                    "Central (P50)": [
                        format_price(x, selected_currency) for x in milestones["Previsao_Central"]
                    ],
                    "Banda 50% (P25–P75)": [
                        f"{format_price(lo, selected_currency)} – {format_price(hi, selected_currency)}"
                        for lo, hi in zip(milestones["Banda_Inf_50"], milestones["Banda_Sup_50"])
                    ],
                    "Banda 80% (P10–P90)": [
                        f"{format_price(lo, selected_currency)} – {format_price(hi, selected_currency)}"
                        for lo, hi in zip(milestones["Banda_Inf_80"], milestones["Banda_Sup_80"])
                    ],
                    "Banda 95% (P2.5–P97.5)": [
                        f"{format_price(lo, selected_currency)} – {format_price(hi, selected_currency)}"
                        for lo, hi in zip(
                            milestones["Banda_Inferior_95"], milestones["Banda_Superior_95"]
                        )
                    ],
                }
            )
            st.dataframe(disp_2y, hide_index=True, use_container_width=True, height=375)

    # ==========================================================================
    # ABA 3: VALIDAÇÃO WALK-FORWARD & AUDITORIA MULTI-JANELA POR REGIMES
    # ==========================================================================
    with tab3:
        st.markdown("### 🔁 Backtesting Walk-Forward Interativo (Janela Única)")
        st.markdown(
            "Escolha uma data de corte no passado. O modelo é treinado **exclusivamente com os dados anteriores ao corte** "
            "(sem vazamento futuro) para projetar 2 anos e comparar com os preços efetivamente observados."
        )

        valid_cutoffs = df_brent.index[252 * 5 : -252]
        default_idx = max(0, len(valid_cutoffs) - 252)
        chosen_cutoff = st.select_slider(
            "Selecione a Data de Corte do Teste Walk-Forward:",
            options=[d.strftime("%Y-%m-%d") for d in valid_cutoffs],
            value=valid_cutoffs[default_idx].strftime("%Y-%m-%d"),
        )

        bt_df, bt_metrics = evaluate_walk_forward_backtest(
            df=df_brent,
            marginal_cost_local=marginal_cost_local,
            cutoff_date_str=chosen_cutoff,
            opec_discipline=0.0,
            geopolitical_risk=0.0,
        )

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("RMSE (Erro Quadrático Médio)", format_price(bt_metrics["rmse"], selected_currency))
        b2.metric("MAPE (Erro Percentual Médio)", f"{bt_metrics['mape']:.2f}%")
        b3.metric("Índice de Acurácia (100 - MAPE)", f"{bt_metrics['accuracy_index']:.1f}%")
        b4.metric(
            "Cobertura Bandas (50% / 80% / 95%)",
            f"{bt_metrics['cov_50']:.0f}% / {bt_metrics['cov_80']:.0f}% / {bt_metrics['band_coverage']:.0f}%",
        )
        b5.metric(
            "Acurácia Direcional",
            f"{bt_metrics['dir_acc']:.1f}%",
            f"Regime: {bt_metrics['regime']}",
        )

        fig_bt = plot_walk_forward_backtest(
            df=df_brent,
            comp_df=bt_df,
            currency_code=selected_currency,
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🏛️ Auditoria Histórica Multi-Janela (Avaliação por Regimes de Mercado)")
        st.markdown(
            "Avaliação automática em **8 janelas deslizantes Out-of-Sample** ao longo da última década, "
            "segmentando o desempenho do modelo entre ciclos de **Alta (Bull)**, **Queda (Bear)** e **Lateralização (Range)**."
        )

        with st.spinner("Executando auditoria multi-janela Walk-Forward..."):
            audit_df = run_multi_window_regime_audit(
                df=df_brent, marginal_cost_local=marginal_cost_local, n_windows=8
            )

        if not audit_df.empty:
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("MAPE Médio Multi-Regime", f"{audit_df['MAPE_Pct'].mean():.2f}%")
            a2.metric("Acurácia Média Global", f"{audit_df['Acuracia_Pct'].mean():.1f}%")
            a3.metric("Cobertura Média Banda 80%", f"{audit_df['Cobertura_IC80_Pct'].mean():.1f}%")
            a4.metric(
                "Acurácia Direcional Média", f"{audit_df['Acuracia_Direcional_Pct'].mean():.1f}%"
            )

            audit_display = pd.DataFrame(
                {
                    "Data de Corte": audit_df["Data_Corte"],
                    "Fim Avaliação": audit_df["Fim_Avaliacao"],
                    "Regime": audit_df["Regime_Mercado"],
                    "Retorno Real (%)": [
                        f"{v:+.1f}%" for v in audit_df["Retorno_Real_Periodo_Pct"]
                    ],
                    "RMSE": [format_price(v, selected_currency) for v in audit_df["RMSE"]],
                    "MAPE (%)": [f"{v:.2f}%" for v in audit_df["MAPE_Pct"]],
                    "Acurácia (%)": [f"{v:.1f}%" for v in audit_df["Acuracia_Pct"]],
                    "Cob. 50% / 80% / 95%": [
                        f"{c50:.0f}% / {c80:.0f}% / {c95:.0f}%"
                        for c50, c80, c95 in zip(
                            audit_df["Cobertura_IC50_Pct"],
                            audit_df["Cobertura_IC80_Pct"],
                            audit_df["Cobertura_IC95_Pct"],
                        )
                    ],
                    "Dir. Acc (%)": [f"{v:.1f}%" for v in audit_df["Acuracia_Direcional_Pct"]],
                }
            )
            st.dataframe(audit_display, hide_index=True, use_container_width=True)

    # ==========================================================================
    # ABA 4: MODELO 2 — LONGO PRAZO ESTRUTURAL (OU + JUMP-DIFFUSION)
    # ==========================================================================
    with tab4:
        st.markdown(
            f"### Projeção Terminal ao Final do Horizonte de {horizon_years} Anos "
            f"({mc_10y_df.index[-1].strftime('%Y')}) — Ornstein-Uhlenbeck + Jump-Diffusion"
        )

        s1, s2, s3, s4 = st.columns(4)
        s1.metric(
            "🟡 Cenário Base / Mediana (P50)",
            format_price(ou_stats["terminal_p50"], selected_currency),
            f"{((ou_stats['terminal_p50'] / current_price) - 1) * 100:+.1f}% vs. Spot",
        )
        s2.metric(
            "🟠 Banda 50% (P25–P75) Terminal",
            f"{format_price(ou_stats['terminal_p25'], selected_currency)} – {format_price(ou_stats['terminal_p75'], selected_currency)}",
            "Faixa Central de Equilíbrio",
        )
        s3.metric(
            "🟢🔴 Banda 80% (P10–P90) Terminal",
            f"{format_price(ou_stats['terminal_p10'], selected_currency)} – {format_price(ou_stats['terminal_p90'], selected_currency)}",
            f"Pessimista P10 / Otimista P90",
        )
        s4.metric(
            "⚪ Banda 95% (P2.5–P97.5) Terminal",
            f"{format_price(ou_stats['terminal_p025'], selected_currency)} – {format_price(ou_stats['terminal_p975'], selected_currency)}",
            f"Custo Marginal: {format_price(marginal_cost_local, selected_currency)}",
        )

        chk_c1, chk_c2 = st.columns(2)
        with chk_c1:
            show_10y_95 = st.checkbox(
                "Exibir faixa externa de 95% (P2.5–P97.5) além das bandas de 50% e 80%",
                value=True,
                key="chk_10y_95",
            )
        with chk_c2:
            show_paths = st.checkbox(
                "Exibir amostra de trajetórias individuais da Simulação de Monte Carlo",
                value=False,
                key="chk_10y_paths",
            )

        fig_10y = plot_structural_monte_carlo_10y(
            df=df_brent,
            mc_df=mc_10y_df,
            ou_stats=ou_stats,
            currency_code=selected_currency,
            show_band_95=show_10y_95,
            show_sample_paths=show_paths,
        )
        st.plotly_chart(fig_10y, use_container_width=True)

        st.markdown("#### 📊 Tabela de Projeção Estrutural Ano a Ano (Bandas 50%, 80% e 95%)")
        yearly_mc = mc_10y_df.iloc[::12].copy()
        yearly_table = pd.DataFrame(
            {
                "Horizonte": [f"Ano {int(a)}" for a in yearly_mc["Ano"]],
                "Data": yearly_mc.index.strftime("%Y-%m-%d"),
                "Extremo Inf. (P2.5)": [
                    format_price(v, selected_currency) for v in yearly_mc["Percentil_025"]
                ],
                "Pessimista (P10)": [
                    format_price(v, selected_currency) for v in yearly_mc["Cenario_Pessimista_P10"]
                ],
                "Banda 50% Inf. (P25)": [
                    format_price(v, selected_currency) for v in yearly_mc["Percentil_25"]
                ],
                "Base / Mediana (P50)": [
                    format_price(v, selected_currency) for v in yearly_mc["Cenario_Base_P50"]
                ],
                "Banda 50% Sup. (P75)": [
                    format_price(v, selected_currency) for v in yearly_mc["Percentil_75"]
                ],
                "Otimista (P90)": [
                    format_price(v, selected_currency) for v in yearly_mc["Cenario_Otimista_P90"]
                ],
                "Extremo Sup. (P97.5)": [
                    format_price(v, selected_currency) for v in yearly_mc["Percentil_975"]
                ],
            }
        )
        st.dataframe(yearly_table, hide_index=True, use_container_width=True)

    # ==========================================================================
    # ABA 5: MÓDULO DE RISCO FINANCEIRO (VaR / CVaR) & SIMULADOR DE HEDGE
    # ==========================================================================
    with tab5:
        st.markdown("### 🛡️ Métricas Institucionais de Risco (VaR & CVaR) — Monte Carlo 5.000 Cenários")
        st.markdown(
            f"Risco extremo calculado diretamente da distribuição estocástica com saltos em **{CURRENCIES[selected_currency]['name']}**:"
        )

        S_mat = ou_stats["sim_matrix"]
        var_horizons = [(1, "1 Mês"), (6, "6 Meses"), (12, "1 Ano"), (24, "2 Anos")]
        var_rows = []
        for m_step, label in var_horizons:
            if m_step < S_mat.shape[1]:
                prices_h = S_mat[:, m_step]
                rel_chg = ((prices_h - current_price) / current_price) * 100.0

                q05_prod = np.percentile(rel_chg, 5)
                q01_prod = np.percentile(rel_chg, 1)
                var95_prod = -q05_prod
                cvar95_prod = -np.mean(rel_chg[rel_chg <= q05_prod])
                var99_prod = -q01_prod

                q95_cons = np.percentile(rel_chg, 95)
                q99_cons = np.percentile(rel_chg, 99)
                var95_cons = q95_cons
                cvar95_cons = np.mean(rel_chg[rel_chg >= q95_cons])
                var99_cons = q99_cons

                var_rows.append(
                    {
                        "Horizonte": label,
                        "Produtor VaR 95% (Queda)": f"-{var95_prod:.1f}%",
                        "Produtor CVaR 95% (Cauda)": f"-{cvar95_prod:.1f}%",
                        "Produtor VaR 99%": f"-{var99_prod:.1f}%",
                        "Consumidor VaR 95% (Alta)": f"+{var95_cons:.1f}%",
                        "Consumidor CVaR 95% (Cauda)": f"+{cvar95_cons:.1f}%",
                        "Consumidor VaR 99%": f"+{var99_cons:.1f}%",
                    }
                )

        st.dataframe(pd.DataFrame(var_rows), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("### ⚖️ Simulador Corporativo de Exposição e Hedge")

        h_col1, h_col2, h_col3, h_col4 = st.columns(4)
        with h_col1:
            corp_profile = st.selectbox(
                "Perfil de Exposição:",
                options=[
                    "Consumidor (Aérea / Transporte / Indústria)",
                    "Produtor (Exploração & Produção)",
                ],
            )
        with h_col2:
            monthly_barrels = st.number_input(
                "Volume Mensal (Barris/mês):",
                min_value=1000,
                max_value=5000000,
                value=50000,
                step=5000,
            )
        with h_col3:
            hedge_months = st.selectbox(
                "Horizonte do Programa de Hedge:",
                options=[12, 24],
                index=0,
                format_func=lambda m: f"{m} Meses ({m * monthly_barrels:,.0f} bbl totais)",
            )
        with h_col4:
            hedge_ratio_pct = st.slider(
                "Percentual Coberto (Hedge Ratio):",
                min_value=0,
                max_value=100,
                value=70,
                step=5,
                format="%d%%",
            )

        strike_price = st.number_input(
            f"Preço Travado / Strike do Hedge ({selected_currency}/bbl):",
            min_value=float(current_price * 0.3),
            max_value=float(current_price * 2.5),
            value=float(round(current_price, 2)),
            step=1.0,
        )

        max_m = min(hedge_months, S_mat.shape[1] - 1)
        sim_prices_window = S_mat[:, 1 : max_m + 1]
        unhedged_cf = np.sum(sim_prices_window * monthly_barrels, axis=1)

        hedge_w = hedge_ratio_pct / 100.0
        effective_prices = (1.0 - hedge_w) * sim_prices_window + hedge_w * strike_price
        hedged_cf = np.sum(effective_prices * monthly_barrels, axis=1)

        is_consumer = corp_profile.startswith("Consumidor")
        p10_unh, p50_unh, p90_unh = np.percentile(unhedged_cf, [10, 50, 90])
        p10_hdg, p50_hdg, p90_hdg = np.percentile(hedged_cf, [10, 50, 90])

        if is_consumer:
            tail_saving = p90_unh - p90_hdg
            tail_label = "Economia no Cenário de Estresse (P90 Alta)"
            base_diff = p50_unh - p50_hdg
        else:
            tail_saving = p10_hdg - p10_unh
            tail_label = "Proteção de Receita no Estresse (P10 Queda)"
            base_diff = p50_hdg - p50_unh

        vol_reduction_pct = (1.0 - (np.std(hedged_cf) / max(np.std(unhedged_cf), 1e-6))) * 100.0

        hm1, hm2, hm3, hm4 = st.columns(4)
        hm1.metric(
            f"{'Custo' if is_consumer else 'Receita'} Base Sem Hedge (P50)",
            format_price(p50_unh, selected_currency),
        )
        hm2.metric(
            f"{'Custo' if is_consumer else 'Receita'} Base Com Hedge (P50)",
            format_price(p50_hdg, selected_currency),
            f"{base_diff:+,.0f} {selected_currency}",
        )
        hm3.metric(
            tail_label,
            format_price(tail_saving, selected_currency),
            f"Hedge de {hedge_ratio_pct}% a {format_price(strike_price, selected_currency)}",
        )
        hm4.metric(
            "Redução da Volatilidade de Caixa",
            f"{vol_reduction_pct:.1f}%",
            f"Exposição: {max_m * monthly_barrels:,.0f} bbl",
        )

        fig_hedge = plot_hedge_distribution(
            unhedged_cashflows=unhedged_cf,
            hedged_cashflows=hedged_cf,
            currency_code=selected_currency,
            profile_label=corp_profile.split(" ")[0],
        )
        st.plotly_chart(fig_hedge, use_container_width=True)

    # ==========================================================================
    # ABA 6: DADOS CONSOLIDADOS & EXPORTAÇÃO CSV
    # ==========================================================================
    with tab6:
        st.markdown("### 💾 Exportação Institucional de Dados Históricos e Projeções")
        st.markdown(
            f"Todos os valores monetários abaixo estão expressos em **{CURRENCIES[selected_currency]['name']}** "
            f"utilizando a série histórica diária exata de câmbio."
        )

        hist_export = df_brent[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "FX_Rate",
                "Close_USD",
                "DXY",
                "WTI_USD",
                "RB_USD",
                "HO_USD",
            ]
        ].copy()
        hist_export["Segmento"] = "Historico_OHLCV_Exogenas"
        hist_export["Moeda"] = selected_currency

        fc2_export = fc_2y_df[
            [
                "Previsao_Central",
                "Banda_Inf_50",
                "Banda_Sup_50",
                "Banda_Inf_80",
                "Banda_Sup_80",
                "Banda_Inferior_95",
                "Banda_Superior_95",
                "Base_Quantitativa_Pura",
            ]
        ].copy()
        fc2_export["Segmento"] = "Modelo_1_Tatico_2Anos"
        fc2_export["Moeda"] = selected_currency

        mc10_export = mc_10y_df[
            [
                "Ano",
                "Percentil_025",
                "Cenario_Pessimista_P10",
                "Percentil_25",
                "Cenario_Base_P50",
                "Percentil_75",
                "Cenario_Otimista_P90",
                "Percentil_975",
                "Media_Monte_Carlo",
            ]
        ].copy()
        mc10_export["Segmento"] = f"Modelo_2_Estrutural_{horizon_years}Anos_JumpDiff"
        mc10_export["Moeda"] = selected_currency

        consolidated_df = pd.concat([hist_export, fc2_export, mc10_export], axis=0)
        consolidated_df.index.name = "Date"

        c_dl1, c_dl2, c_dl3, c_dl4 = st.columns(4)
        with c_dl1:
            st.download_button(
                label="⬇️ Baixar Dataset Completo (CSV)",
                data=consolidated_df.to_csv().encode("utf-8"),
                file_name=f"brent_quant_completo_{selected_currency}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c_dl2:
            st.download_button(
                label="⬇️ Histórico OHLCV + FX + Macro (CSV)",
                data=hist_export.to_csv().encode("utf-8"),
                file_name=f"brent_historico_ohlcv_{selected_currency}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c_dl3:
            st.download_button(
                label="⬇️ Projeção Tática 2 Anos (CSV)",
                data=fc2_export.to_csv().encode("utf-8"),
                file_name=f"brent_projecao_2anos_{selected_currency}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c_dl4:
            st.download_button(
                label=f"⬇️ Projeção Estrutural {horizon_years}A (CSV)",
                data=mc10_export.to_csv().encode("utf-8"),
                file_name=f"brent_projecao_{horizon_years}anos_ou_{selected_currency}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("#### Pré-visualização dos Últimos Registros Históricos (OHLCV + Câmbio + Exógenas)")
        st.dataframe(hist_export.tail(20).sort_index(ascending=False), use_container_width=True)


if __name__ == "__main__":
    main()
