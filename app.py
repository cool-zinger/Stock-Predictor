"""MarketScope: daily stock research, delayed quotes, and direct ML forecasts.

Python 3.12+. Install requirements.txt, then run: streamlit run app.py
All application logic is contained in this file. No API key is required.

API references:
https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data
https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

LOGGER = logging.getLogger("marketscope")
FEATURES = ["Close", "Volume", "SMA_20", "SMA_50", "RSI_14", "MACD"]
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
RISK_NOTICE = (
    "Stock market prediction carries inherent risk, and historical performance "
    "does not guarantee future results."
)
MODEL_VERSION = "1.0.0"
COVERAGE = 0.90
MIN_SAMPLES = 320
FOREST_TREES = 350
COLORS = {
    "price": "#38BDF8",
    "fast": "#FBBF24",
    "slow": "#A78BFA",
    "forecast": "#2DD4BF",
    "negative": "#FB7185",
    "muted": "#94A3B8",
}
# Yahoo exchange identifiers, not an assumption based on the ticker's spelling.
EXCHANGE_CALENDARS = {
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "NCM": "NASDAQ",
    "NAS": "NASDAQ",
    "NYQ": "NYSE",
    "ASE": "NYSE",
    "PCX": "NYSE",
    "BTS": "NYSE",
    "BATS": "NYSE",
    "LSE": "LSE",
    "TOR": "TSX",
    "HKG": "HKEX",
    "JPX": "JPX",
    "ASX": "ASX",
    "NSI": "NSE",
    "BSE": "BSE",
    "GER": "XETR",
    "PAR": "XPAR",
    "AMS": "XAMS",
    "EBS": "SIX",
}


class DataUnavailable(RuntimeError):
    """A recoverable market-data failure with a user-facing explanation."""


class InsufficientData(ValueError):
    """The requested sample cannot support the forecasting protocol."""


def validate_ticker(value: str) -> str:
    symbol = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9^][A-Z0-9.^=\-]{0,24}", symbol):
        raise ValueError(
            "Enter one valid Yahoo Finance symbol, such as AAPL or RELIANCE.NS."
        )
    return symbol


def clean_bars(raw: pd.DataFrame, *, daily: bool = True) -> pd.DataFrame:
    """Keep missing feature inputs missing; never forward/back-fill market data."""
    if raw is None or raw.empty:
        raise DataUnavailable(
            "No prices were returned. Check the symbol and date range."
        )
    if not all(column in raw.columns for column in OHLCV):
        raise DataUnavailable(
            "The provider returned an incomplete price table. Try again later."
        )
    data = raw.loc[:, OHLCV].copy()
    data.index = pd.DatetimeIndex(pd.to_datetime(data.index))
    data = data.loc[~data.index.isna()].sort_index()
    data = data.loc[~data.index.duplicated(keep="last")]
    for column in OHLCV:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    # Missing/negative volume remains NaN and is excluded from model inputs later.
    data.loc[data["Volume"] < 0, "Volume"] = np.nan
    for column in ["Open", "High", "Low", "Close"]:
        data.loc[data[column] <= 0, column] = np.nan
    data = data.dropna(subset=["Close"])
    if daily:
        # Remove timezone only after preserving the exchange's local session date.
        data.index = data.index.tz_localize(None).normalize()
        data = data.loc[~data.index.duplicated(keep="last")]
    data.index.name = "Date"
    if data.empty:
        raise DataUnavailable("The provider returned no usable, positive prices.")
    return data


def request_history(symbol: str, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Bounded requests with one retry; exceptions are not cached as successful data."""
    for attempt in range(2):
        try:
            ticker = yf.Ticker(symbol)
            raw = ticker.history(timeout=15, actions=False, **kwargs)
            if raw is None or raw.empty:
                raise DataUnavailable("No data returned")
            # history() already populated this metadata, so no .info request is needed.
            try:
                metadata = dict(ticker.history_metadata or {})
            except Exception:
                LOGGER.debug("Optional quote metadata unavailable", exc_info=True)
                metadata = {}
            if isinstance(raw.index, pd.DatetimeIndex) and raw.index.tz is not None:
                metadata.setdefault("exchangeTimezoneName", str(raw.index.tz))
            return raw, metadata
        except Exception as exc:
            LOGGER.warning(
                "History request failed for %s (attempt %s): %s",
                symbol,
                attempt + 1,
                type(exc).__name__,
            )
            if attempt == 1:
                raise DataUnavailable(
                    f"Prices for {symbol} are unavailable. The symbol may be invalid, "
                    "the market-data service may be busy, or the request timed out. "
                    "Check your connection and retry."
                ) from exc
            time.sleep(0.75)
    raise DataUnavailable("Market data unavailable")


def local_today(metadata: dict[str, Any]) -> date:
    try:
        zone = ZoneInfo(str(metadata.get("exchangeTimezoneName", "UTC")))
    except (ZoneInfoNotFoundError, ValueError):
        zone = timezone.utc
    return datetime.now(zone).date()


@st.cache_data(ttl=3600, max_entries=32, show_spinner=False)
def fetch_history(symbol: str, start: date, end: date) -> dict[str, Any]:
    """Fetch extra past bars for indicator warm-up; end is inclusive in the UI."""
    raw, metadata = request_history(
        symbol,
        start=(start - timedelta(days=400)).isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=True,
    )
    bars = clean_bars(raw)
    # Conservatively exclude today's potentially incomplete bar, even after close.
    bars = bars.loc[bars.index.date < local_today(metadata)]
    if bars.empty:
        raise DataUnavailable("No completed daily bars are available in this range.")
    return {
        "bars": bars,
        "metadata": metadata,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


@st.cache_data(ttl=300, max_entries=64, show_spinner=False)
def fetch_quote_daily(symbol: str) -> dict[str, Any]:
    raw, metadata = request_history(
        symbol, period="2y", interval="1d", auto_adjust=False
    )
    return {"bars": clean_bars(raw), "metadata": metadata}


@st.cache_data(ttl=60, max_entries=64, show_spinner=False)
def fetch_quote(symbol: str) -> dict[str, Any]:
    """Use the latest minute close, falling back explicitly to daily data."""
    daily_payload = fetch_quote_daily(symbol)
    daily = daily_payload["bars"]
    metadata = daily_payload["metadata"]
    source = "Latest daily bar"
    stamp = daily.index[-1]
    price = float(daily["Close"].iloc[-1])
    try:
        raw, minute_meta = request_history(
            symbol, period="5d", interval="1m", auto_adjust=False
        )
        minute = clean_bars(raw, daily=False)
        if minute.index[-1].date() >= stamp.date():
            stamp = minute.index[-1]
            price = float(minute["Close"].iloc[-1])
            metadata = {**metadata, **minute_meta}
            source = "Latest 1-minute bar"
    except DataUnavailable:
        LOGGER.info("Using daily quote fallback for %s", symbol)
    previous = daily.loc[daily.index.date < stamp.date(), "Close"]
    previous_close = float(previous.iloc[-1]) if not previous.empty else np.nan
    # Anchor to today's date, never to an old user-selected historical end date.
    cutoff = pd.Timestamp(local_today(metadata)) - pd.Timedelta(weeks=52)
    annual = daily.loc[daily.index >= cutoff]
    high = float(annual["High"].max()) if not annual.empty else np.nan
    low = float(annual["Low"].min()) if not annual.empty else np.nan
    if pd.Timestamp(stamp.date()) >= cutoff:
        high = max(high, price) if np.isfinite(high) else price
        low = min(low, price) if np.isfinite(low) else price
    return {
        "price": price,
        "change": 100 * (price / previous_close - 1),
        "high": high,
        "low": low,
        "timestamp": stamp.isoformat(),
        "source": source,
        "currency": metadata.get("currency", "quote units"),
        "partial_year": daily.index[0] > cutoff + pd.Timedelta(days=7),
        "stale": (local_today(metadata) - stamp.date()).days > 4,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder smoothing seeded with the first period's simple gain/loss average."""
    changes = close.diff().to_numpy(dtype=float)
    values = np.full(len(close), np.nan)
    if len(close) <= period:
        return pd.Series(values, index=close.index, name=f"RSI_{period}")
    gains = np.maximum(changes, 0.0)
    losses = np.maximum(-changes, 0.0)
    avg_gain = float(np.mean(gains[1 : period + 1]))
    avg_loss = float(np.mean(losses[1 : period + 1]))
    for i in range(period, len(close)):
        if i > period:
            # Wilder's recursive mean: A_t = ((n-1)*A_(t-1) + x_t) / n.
            avg_gain = ((period - 1) * avg_gain + gains[i]) / period
            avg_loss = ((period - 1) * avg_loss + losses[i]) / period
        # RSI = 100 - 100/(1 + average_gain/average_loss). A flat series is neutral.
        if avg_gain == 0 and avg_loss == 0:
            values[i] = 50.0
        elif avg_loss == 0:
            values[i] = 100.0
        else:
            values[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return pd.Series(values, index=close.index, name=f"RSI_{period}")


def engineer_features(bars: pd.DataFrame) -> pd.DataFrame:
    data = bars.copy()
    close = data["Close"]
    data["SMA_20"] = close.rolling(20, min_periods=20).mean()
    data["SMA_50"] = close.rolling(50, min_periods=50).mean()
    data["EMA_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    data["RSI_14"] = wilder_rsi(close)
    fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    data["MACD"] = fast - slow
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]
    return data.replace([np.inf, -np.inf], np.nan)


def supervised_samples(
    data: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, list[str]]:
    """Shift on the session index BEFORE dropping missing indicator/volume rows."""
    if not 1 <= horizon <= 30:
        raise ValueError("Forecast horizon must be between 1 and 30 trading sessions.")
    labels = [f"Close_t+{step}" for step in range(1, horizon + 1)]
    frame = data.loc[:, FEATURES].copy()
    for step, column in enumerate(labels, 1):
        frame[column] = data["Close"].shift(-step)
    frame["Target_Date"] = pd.Series(data.index, index=data.index).shift(-horizon)
    return frame.replace([np.inf, -np.inf], np.nan).dropna(), labels


def chronological_holdout(
    frame: pd.DataFrame, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    split = int(len(frame) * 0.80)
    # The last 20% is untouched test data. Purge H origins from the training end:
    # their shifted labels could otherwise reach into the first test origin/date.
    train = np.arange(max(0, split - horizon))
    test = np.arange(split, len(frame))
    if len(train) < 180 or len(test) < 40:
        raise InsufficientData("Select a longer date range (at least about two years).")
    if frame["Target_Date"].iloc[train].max() >= frame.index[test[0]]:
        raise ValueError("Training label dates overlap the test period.")
    return train, test


def new_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=FOREST_TREES,
        max_depth=10,
        min_samples_leaf=5,
        max_features=1.0,
        bootstrap=True,
        random_state=42,
        n_jobs=max(1, min(4, os.cpu_count() or 1)),
    )


def fit_forest(x: pd.DataFrame, y: pd.DataFrame) -> RandomForestRegressor:
    model = new_forest()
    model.fit(x, y.iloc[:, 0] if y.shape[1] == 1 else y)
    return model


def predict_matrix(model: RandomForestRegressor, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(x)).reshape(len(x), -1)


@st.cache_data(ttl=3600, max_entries=16, show_spinner=False)
def run_forecast(
    data: pd.DataFrame, horizon: int, version: str = MODEL_VERSION
) -> dict[str, Any]:
    """Direct multi-output forest; no invented future volume or recursive indicators."""
    frame, labels = supervised_samples(data, horizon)
    if len(frame) < MIN_SAMPLES:
        raise InsufficientData(
            f"Only {len(frame):,} usable labeled sessions; at least {MIN_SAMPLES} are "
            "required for validation. Extend the date range or reduce the horizon."
        )
    latest = data.loc[:, FEATURES].iloc[[-1]]
    if not np.isfinite(latest.to_numpy(dtype=float)).all():
        raise InsufficientData(
            "The latest bar has incomplete features. Try another range or refresh data."
        )
    train_idx, test_idx = chronological_holdout(frame, horizon)
    train, test = frame.iloc[train_idx], frame.iloc[test_idx]
    # Expanding-window calibration uses ONLY the training partition. An H-row
    # gap in every fold also prevents future target labels crossing its boundary.
    fold_size = min(100, (len(train) - horizon - 80) // 3)
    if fold_size < 20:
        raise InsufficientData(
            "Too few observations for interval calibration. Extend the range."
        )
    folds = TimeSeriesSplit(n_splits=3, test_size=fold_size, gap=horizon)
    errors, fold_mae, fold_audit = [], [], []
    for fit_idx, cal_idx in folds.split(train):
        fit, cal = train.iloc[fit_idx], train.iloc[cal_idx]
        if fit["Target_Date"].max() >= cal.index[0]:
            raise ValueError("Calibration label dates overlap a validation origin.")
        model = fit_forest(fit[FEATURES], fit[labels])
        prediction = predict_matrix(model, cal[FEATURES])
        # Normalize by the known origin price so widths adapt to the current scale.
        errors.append(
            np.abs(cal[labels].to_numpy() - prediction) / cal[["Close"]].to_numpy()
        )
        fold_mae.append(mean_absolute_error(cal[labels[-1]], prediction[:, -1]))
        fold_audit.append(
            {
                "train_target_end": fit["Target_Date"].max(),
                "validation_origin_start": cal.index[0],
            }
        )
    calibration = np.vstack(errors)
    # Finite-sample order statistic. Serial dependence means this remains an
    # empirical pointwise prediction band, NOT a guaranteed confidence interval.
    rank = min(len(calibration), math.ceil((len(calibration) + 1) * COVERAGE))
    widths = np.sort(calibration, axis=0)[rank - 1]

    evaluation_model = fit_forest(train[FEATURES], train[labels])
    predicted = predict_matrix(evaluation_model, test[FEATURES])
    actual = test[labels[-1]].to_numpy()
    terminal = predicted[:, -1]
    naive = test["Close"].to_numpy()
    lower = np.maximum(0, terminal - widths[-1] * naive)
    upper = terminal + widths[-1] * naive
    r2 = float(r2_score(actual, terminal)) if np.var(actual) > 1e-12 else np.nan
    test_table = pd.DataFrame(
        {
            "Origin date": test.index,
            "Target date": test["Target_Date"].to_numpy(),
            "Actual": actual,
            "Predicted": terminal,
            "Unchanged-price baseline": naive,
            "Lower 90%": lower,
            "Upper 90%": upper,
        }
    ).set_index("Target date")
    # Refit separately on all matured labels, including the former test period,
    # only AFTER its predictions/metrics have been frozen. Latest H rows have no
    # fully observed target vector, but the latest feature row is used at inference.
    final_model = fit_forest(frame[FEATURES], frame[labels])
    path = predict_matrix(final_model, latest)[0]
    anchor = float(latest["Close"].iloc[0])
    forecast = pd.DataFrame(
        {
            "Session ahead": np.arange(1, horizon + 1),
            "Forecast": path,
            "Lower 90%": np.maximum(0, path - widths * anchor),
            "Upper 90%": path + widths * anchor,
        }
    )
    return {
        "forecast": forecast,
        "test": test_table,
        "mae": float(mean_absolute_error(actual, terminal)),
        "r2": r2,
        "baseline_mae": float(mean_absolute_error(actual, naive)),
        "coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "cv_mae": float(np.mean(fold_mae)),
        "calibration_count": len(calibration),
        "train_count": len(train),
        "test_count": len(test),
        "purged_count": horizon,
        "train_start": train.index[0],
        "train_end": train.index[-1],
        "train_target_end": train["Target_Date"].max(),
        "test_origin_start": test.index[0],
        "asof": data.index[-1],
        "anchor": anchor,
        "labeled_count": len(frame),
        "fold_audit": fold_audit,
        "version": version,
        "importance": pd.Series(
            evaluation_model.feature_importances_, index=FEATURES
        ).sort_values(),
    }


@st.cache_data(ttl=86400, max_entries=128, show_spinner=False)
def future_sessions(
    asof: date, horizon: int, exchange: str
) -> tuple[pd.DatetimeIndex, str]:
    name = EXCHANGE_CALENDARS.get(exchange)
    if name:
        try:
            schedule = mcal.get_calendar(name).schedule(
                start_date=asof + timedelta(days=1),
                end_date=asof + timedelta(days=120),
            )
            sessions = pd.DatetimeIndex(schedule.index).tz_localize(None)
            if len(sessions) >= horizon:
                return sessions[:horizon], f"{name} trading calendar"
        except Exception:
            LOGGER.warning("Calendar unavailable for %s", exchange, exc_info=True)
    dates = pd.bdate_range(pd.Timestamp(asof) + pd.offsets.BDay(1), periods=horizon)
    return (
        dates,
        "Estimated weekdays; exchange holidays are not available for this symbol",
    )


def chart_palette(dark: bool) -> dict[str, str]:
    if dark:
        return COLORS.copy()
    return {
        "price": "#0369A1",
        "fast": "#B45309",
        "slow": "#7C3AED",
        "forecast": "#0F766E",
        "negative": "#BE123C",
        "muted": "#64748B",
    }


def style_chart(fig: go.Figure, dark: bool, height: int = 440) -> go.Figure:
    background = "#101827" if dark else "#FFFFFF"
    grid = "#243044" if dark else "#E8EDF3"
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        height=height,
        paper_bgcolor=background,
        plot_bgcolor=background,
        font={
            "family": "Arial, sans-serif",
            "size": 12,
            "color": "#DCE6F2" if dark else "#334155",
        },
        margin={"l": 15, "r": 20, "t": 45, "b": 20},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0, "font": {"size": 11}},
        hoverlabel={"font_size": 12},
    )
    fig.update_xaxes(showgrid=False, zeroline=False, rangeslider_visible=False)
    fig.update_yaxes(gridcolor=grid, zeroline=False, tickformat=",.2f")
    return fig


def line_trace(x: Any, y: Any, name: str, color: str, **kwargs: Any) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        name=name,
        mode="lines",
        line={"color": color, "width": 2},
        hovertemplate="%{y:,.2f}<extra>%{fullData.name}</extra>",
        **kwargs,
    )


def historical_chart(data: pd.DataFrame, dark: bool, currency: str) -> go.Figure:
    palette = chart_palette(dark)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.8, 0.2],
        vertical_spacing=0.035,
    )
    for field, label, color in [
        ("Close", "Adjusted close", "price"),
        ("SMA_20", "SMA 20", "fast"),
        ("SMA_50", "SMA 50", "slow"),
    ]:
        fig.add_trace(
            line_trace(data.index, data[field], label, palette[color]), row=1, col=1
        )
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data["Volume"],
            name="Volume",
            marker_color=palette["muted"],
            opacity=0.4,
            showlegend=False,
            hovertemplate="Volume %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    style_chart(fig, dark, 510)
    fig.update_yaxes(title_text=currency, row=1, col=1)
    fig.update_yaxes(title_text="Volume", tickformat=".2s", row=2, col=1)
    fig.update_layout(uirevision="historical")
    return fig


def validation_chart(result: dict[str, Any], dark: bool, currency: str) -> go.Figure:
    palette = chart_palette(dark)
    table = result["test"]
    fig = go.Figure()
    for field, color in [
        ("Actual", "price"),
        ("Predicted", "forecast"),
        ("Unchanged-price baseline", "muted"),
    ]:
        trace = line_trace(table.index, table[field], field, palette[color])
        if field == "Unchanged-price baseline":
            trace.update(line_dash="dot", visible="legendonly")
        fig.add_trace(trace)
    style_chart(fig, dark)
    fig.update_yaxes(title_text=currency)
    fig.update_xaxes(title_text="Target session date")
    return fig


def forecast_chart(
    data: pd.DataFrame, forecast: pd.DataFrame, dark: bool, currency: str
) -> go.Figure:
    palette = chart_palette(dark)
    recent = data.tail(65)
    anchor = float(recent["Close"].iloc[-1])
    x = pd.DatetimeIndex([data.index[-1], *forecast.index])
    # The observed anchor has zero forecast uncertainty. H=1 still draws a band.
    lower = np.r_[anchor, forecast["Lower 90%"].to_numpy()]
    upper = np.r_[anchor, forecast["Upper 90%"].to_numpy()]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=lower, mode="lines", line_width=0, hoverinfo="skip", showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=upper,
            mode="lines",
            line_width=0,
            fill="tonexty",
            fillcolor="rgba(20,184,166,0.18)",
            name="90% empirical band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        line_trace(recent.index, recent["Close"], "Observed close", palette["price"])
    )
    prediction = line_trace(
        x, np.r_[anchor, forecast["Forecast"]], "Direct forecast", palette["forecast"]
    )
    prediction.update(mode="lines+markers", marker_size=5, line_dash="dash")
    fig.add_trace(prediction)
    fig.add_vline(
        x=data.index[-1].to_pydatetime(), line_dash="dot", line_color=palette["muted"]
    )
    style_chart(fig, dark)
    fig.update_yaxes(title_text=currency)
    return fig


def indicator_chart(data: pd.DataFrame, dark: bool, currency: str) -> go.Figure:
    palette = chart_palette(dark)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.42, 0.28, 0.30],
        vertical_spacing=0.065,
    )
    for field, label, color in [
        ("Close", "Adjusted close", "price"),
        ("EMA_20", "EMA 20", "fast"),
    ]:
        fig.add_trace(
            line_trace(data.index, data[field], label, palette[color]), row=1, col=1
        )
    fig.add_trace(
        line_trace(data.index, data["RSI_14"], "RSI 14", palette["slow"]), row=2, col=1
    )
    fig.add_hrect(
        y0=30,
        y1=70,
        fillcolor=palette["muted"],
        opacity=0.08,
        line_width=0,
        row=2,
        col=1,
    )
    for threshold in [30, 70]:
        fig.add_hline(
            y=threshold, line_dash="dot", line_color=palette["muted"], row=2, col=1
        )
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data["MACD_Hist"],
            name="MACD histogram",
            marker_color=np.where(
                data["MACD_Hist"] >= 0, palette["forecast"], palette["negative"]
            ),
            opacity=0.45,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        line_trace(data.index, data["MACD"], "MACD", palette["price"]), row=3, col=1
    )
    fig.add_trace(
        line_trace(data.index, data["MACD_Signal"], "Signal 9", palette["fast"]),
        row=3,
        col=1,
    )
    style_chart(fig, dark, 700)
    fig.update_yaxes(title_text=currency, row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_layout(legend_y=1.07)
    return fig


def show_chart(fig: go.Figure, key: str) -> None:
    st.plotly_chart(
        fig,
        theme=None,
        width="stretch",
        key=key,
        config={
            "displaylogo": False,
            "scrollZoom": False,
            "toImageButtonOptions": {"format": "png", "scale": 2},
        },
    )


def number(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}" if np.isfinite(value) else "N/A"


@st.fragment(run_every="60s")
def render_quote_strip(symbol: str) -> None:
    try:
        with st.spinner("Updating market snapshot…"):
            quote = fetch_quote(symbol)
    except DataUnavailable as exc:
        st.warning(f"Live snapshot unavailable. {exc}")
        st.caption("Historical analysis can still be used below.")
        return
    with st.container(horizontal=True):
        st.metric(
            "Current Price",
            number(quote["price"]),
            border=True,
            help=f"{quote['currency']}; latest available Yahoo price, potentially delayed.",
        )
        st.metric(
            "Daily Change (%)",
            f"{number(quote['change'])}%",
            border=True,
            help="Relative to the previous daily close before the quoted session.",
        )
        st.metric(
            "52-Week High",
            number(quote["high"]),
            border=True,
            help="Highest daily High in the trailing 52 weeks; excludes dividend adjustments.",
        )
        st.metric(
            "52-Week Low",
            number(quote["low"]),
            border=True,
            help="Lowest daily Low in the trailing 52 weeks; excludes dividend adjustments.",
        )
    st.caption(
        f"{quote['currency']} · {quote['source']} · Quote timestamp: {quote['timestamp']} · "
        f"Checked {quote['fetched_at']}. Refreshes every 60 seconds while this page is open. "
        "Yahoo data may be delayed; this is not an exchange real-time feed."
    )
    if quote["stale"]:
        st.warning(
            "The latest quote is more than four calendar days old. Check the timestamp before using it."
        )
    if quote["partial_year"]:
        st.caption(
            "52-week extremes use the available history, which covers less than a full year."
        )


def render_history(data: pd.DataFrame, symbol: str, dark: bool, currency: str) -> None:
    with st.container(border=True):
        st.subheader("Price & moving averages")
        st.caption("Daily adjusted prices · 20 / 50-session moving averages · volume")
        show_chart(historical_chart(data, dark, currency), "history_chart")
    returns = data["Close"].pct_change().dropna()
    total_return = 100 * (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1)
    drawdown = 100 * (data["Close"] / data["Close"].cummax() - 1).min()
    volatility = 100 * returns.std() * np.sqrt(252) if len(returns) > 1 else np.nan
    with st.container(horizontal=True):
        st.metric("Period change", f"{total_return:+.2f}%", border=True)
        st.metric(
            "Annualized volatility",
            f"{number(volatility)}%",
            border=True,
            help="Sample standard deviation of daily returns × √252; a convention for equities.",
        )
        st.metric("Maximum drawdown", f"{drawdown:.2f}%", border=True)
        st.metric("Daily observations", f"{len(data):,}", border=True)
    with st.expander("Explore & export historical data"):
        st.dataframe(
            data.sort_index(ascending=False),
            width="stretch",
            height=350,
            column_config={
                column: st.column_config.NumberColumn(format="%.2f")
                for column in data.columns
                if column != "Volume"
            },
        )
        st.download_button(
            "Download history CSV",
            data.to_csv().encode("utf-8"),
            file_name=f"{symbol}_history.csv",
            mime="text/csv",
            icon=":material/download:",
        )


def render_models(
    data: pd.DataFrame,
    horizon: int,
    symbol: str,
    metadata: dict[str, Any],
    dark: bool,
    currency: str,
) -> None:
    st.warning(RISK_NOTICE, icon=":material/warning:")
    with st.spinner("Fitting the forest and evaluating chronological holdouts…"):
        try:
            result = run_forecast(data, horizon)
        except InsufficientData as exc:
            st.info(str(exc))
            return
        except Exception:
            LOGGER.exception("Model computation failed for %s", symbol)
            st.error(
                "The model could not be fitted to this data. Refresh prices or select a longer range."
            )
            return
    dates, calendar_note = future_sessions(
        result["asof"].date(), horizon, str(metadata.get("exchangeName", ""))
    )
    forecast = result["forecast"].copy()
    forecast.index = dates
    forecast.index.name = "Date"
    with st.container(border=True):
        st.subheader(f"Model evaluation · {horizon}-session horizon")
        st.caption("Random forest · Fixed chronological holdout · Lower MAE is better")
        with st.container(horizontal=True):
            st.metric(
                "Mean absolute error",
                number(result["mae"]),
                help=f"Average absolute forecast error on the held-out period, in {currency}.",
            )
            st.metric(
                "R² score",
                number(result["r2"], 3),
                help="Can be negative; 1 is perfect. Not a probability or accuracy percentage.",
            )
            st.metric(
                "Unchanged-price MAE",
                number(result["baseline_mae"]),
                help="Forecasts the origin day's close for the target session.",
            )
            st.metric(
                "Test interval coverage",
                f"{result['coverage']:.1%}",
                help="Observed coverage of the nominal 90% pointwise band on untouched test targets.",
            )
        st.caption(
            f"80% / 20% chronological allocation before a {horizon}-session boundary purge. "
            f"Effective training: {result['train_count']:,} · Test: {result['test_count']:,} · "
            f"Calibration: {result['calibration_count']:,} earlier observations."
        )
    if result["mae"] >= result["baseline_mae"]:
        st.warning(
            "The model did not outperform the unchanged-price baseline on this test period."
        )
    if result["coverage"] < 0.80:
        st.warning(
            "Test interval coverage is below 80%; the displayed uncertainty band may be too narrow."
        )
    with st.container(border=True):
        st.subheader("Actual vs. predicted")
        st.caption(
            "Each prediction uses features observed at its origin, before the plotted target date. "
            "The evaluation model remains fixed throughout this test period."
        )
        show_chart(validation_chart(result, dark, currency), "validation_chart")
    with st.container(border=True):
        st.subheader("Future forecast timeline")
        st.caption(
            f"Forecast origin: {result['asof']:%d %b %Y} · {calendar_note} · Adjusted-price basis"
        )
        last = forecast.iloc[-1]
        with st.container(horizontal=True):
            st.metric(
                f"Projected close · {dates[-1]:%d %b}",
                number(last["Forecast"]),
                delta=f"{100 * (last['Forecast'] / result['anchor'] - 1):+.2f}% vs. origin",
                delta_color="off",
            )
            st.metric(
                "90% empirical range",
                f"{number(last['Lower 90%'])} – {number(last['Upper 90%'])}",
            )
        show_chart(forecast_chart(data, forecast, dark, currency), "forecast_chart")
        st.caption(
            "The band uses earlier expanding-window forecast errors, scaled by origin price. "
            "It is a pointwise empirical prediction interval, not a guaranteed confidence interval "
            "or a 90% guarantee for the whole path. Regime changes and serial dependence can reduce coverage."
        )
        if result["asof"].date() < local_today(metadata) - timedelta(days=7):
            st.info(
                "This forecast starts at your historical range's last available bar, not at today's quote."
            )
    with st.expander("Methodology & validation audit"):
        st.markdown(
            "The forest jointly predicts **Close(t+1) … Close(t+H)** using Close, Volume, "
            "SMA 20, SMA 50, RSI 14, and MACD. The reported MAE and R² evaluate **Close(t+H)**. "
            "EMA 20 is displayed as an indicator. No future features are invented.\n\n"
            "Three expanding training folds calibrate uncertainty, each with a horizon-sized gap. "
            "The last 20% of labeled observations are reserved for evaluation. Training origins "
            "whose targets could reach the test period are removed. Parameters are fixed in advance. "
            "After evaluation, a separate forest is fitted to all observed target vectors for the current forecast.\n\n"
            "Tree models cannot extrapolate reliably beyond training price levels. A high R² on price levels "
            "does not establish tradable returns; use the baseline comparison. Evaluation excludes "
            "transaction costs, execution, and trading decisions. Yahoo adjusted histories can be revised "
            "for corporate actions and are not point-in-time archived data."
        )
        st.caption(
            f"Last training target: {result['train_target_end']:%Y-%m-%d} · "
            f"First test origin: {result['test_origin_start']:%Y-%m-%d} · "
            f"Mean training-fold MAE: {result['cv_mae']:.2f} {currency} · "
            f"{FOREST_TREES} trees · Seed 42 · Model {result['version']}"
        )
        importance = (
            result["importance"]
            .rename("Relative importance")
            .sort_values(ascending=False)
        )
        st.dataframe(
            importance,
            column_config={
                "Relative importance": st.column_config.ProgressColumn(
                    format="percent", min_value=0, max_value=1
                ),
            },
        )
        st.caption(
            "Impurity-based feature importance is descriptive and can be biased by correlated indicators."
        )
    with st.expander("Forecast table & downloads"):
        st.dataframe(
            forecast,
            column_config={
                col: st.column_config.NumberColumn(format="%.2f")
                for col in ["Forecast", "Lower 90%", "Upper 90%"]
            },
        )
        with st.container(horizontal=True):
            st.download_button(
                "Download forecast CSV",
                forecast.to_csv().encode("utf-8"),
                f"{symbol}_{horizon}_session_forecast.csv",
                "text/csv",
            )
            st.download_button(
                "Download test predictions",
                result["test"].to_csv().encode("utf-8"),
                f"{symbol}_{horizon}_session_test.csv",
                "text/csv",
            )


def render_indicators(data: pd.DataFrame, dark: bool, currency: str) -> None:
    latest = data.iloc[-1]
    with st.container(horizontal=True):
        st.metric(
            "RSI · 14",
            number(latest["RSI_14"]),
            border=True,
            help="Wilder-smoothed momentum, 0–100. Common reference levels: 30 and 70.",
        )
        st.metric("EMA · 20", number(latest["EMA_20"]), border=True)
        st.metric("MACD · 12 / 26", number(latest["MACD"]), border=True)
        st.metric("Signal · 9", number(latest["MACD_Signal"]), border=True)
    window = st.selectbox(
        "Indicator chart window",
        ["6 months", "1 year", "Full range"],
        key="indicator_window",
    )
    days = {"6 months": 183, "1 year": 366}.get(window)
    selected = (
        data.loc[data.index >= data.index[-1] - pd.Timedelta(days=days)]
        if days
        else data
    )
    with st.container(border=True):
        show_chart(indicator_chart(selected, dark, currency), "indicators_chart")
    st.caption(
        "Indicators describe past price behavior. RSI thresholds and MACD crossings are not standalone trading signals."
    )


def main() -> None:
    st.set_page_config(
        page_title="MarketScope | Stock research",
        page_icon=":material/monitoring:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    today = datetime.now(timezone.utc).date()
    with st.sidebar:
        st.title("MarketScope")
        st.caption("EQUITY RESEARCH WORKSPACE")
        symbol_input = st.text_input(
            "Ticker symbol", "AAPL", max_chars=25, key="ticker"
        )
        selected_dates = st.date_input(
            "Date range",
            value=((pd.Timestamp(today) - pd.DateOffset(years=5)).date(), today),
            min_value=date(1980, 1, 1),
            max_value=today,
            key="date_range",
        )
        horizon = st.slider(
            "Forecast Horizon",
            1,
            30,
            10,
            key="horizon",
            help="Trading sessions, not calendar days. Market holidays are skipped when known.",
        )
        chart_theme = st.selectbox(
            "Chart appearance", ["Match app", "Dark", "Light"], key="chart_theme"
        )
        refresh = st.button(
            "Refresh market data", icon=":material/refresh:", width="stretch"
        )
        st.caption("Daily research · Intraday snapshot · 1–30 trading sessions")
        st.caption(
            "Switch the app's light/dark appearance in Settings. Quotes update independently of the forecast."
        )
    st.title("Stock research & forecasting")
    st.caption(
        "Historical context, transparent model evaluation, and technical indicators in one workspace."
    )
    try:
        symbol = validate_ticker(symbol_input)
    except ValueError as exc:
        st.error(str(exc))
        return
    if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
        st.info("Select both a start and an end date to load the dashboard.")
        return
    start, end = selected_dates
    if start >= end:
        st.error("The start date must be earlier than the end date.")
        return
    if refresh:
        # Invalidate only the requested symbol/range, not every user's cached data.
        fetch_history.clear(symbol, start, end)
        fetch_quote.clear(symbol)
        fetch_quote_daily.clear(symbol)
    dark = chart_theme == "Dark" or (
        chart_theme == "Match app" and st.context.theme.type == "dark"
    )
    st.subheader(symbol)
    render_quote_strip(symbol)
    with st.spinner("Loading completed daily prices…"):
        try:
            payload = fetch_history(symbol, start, end)
        except DataUnavailable as exc:
            st.error(str(exc))
            return
    engineered = engineer_features(payload["bars"])
    data = engineered.loc[
        (engineered.index.date >= start) & (engineered.index.date <= end)
    ]
    if data.empty:
        st.info(
            "No completed daily bars fall inside this date range. Extend the range."
        )
        return
    metadata = payload["metadata"]
    currency = str(metadata.get("currency", "quote units"))
    st.caption(
        f"Research window: {data.index[0]:%d %b %Y} — {data.index[-1]:%d %b %Y} · "
        f"{currency} · Split/dividend-adjusted history. Models conservatively exclude today's "
        "exchange-local daily bar; the live snapshot above uses unadjusted dividend prices."
    )
    missing = int(data[FEATURES].isna().any(axis=1).sum())
    if missing:
        st.caption(
            f"{missing} rows contain incomplete indicator or volume inputs and will be excluded from fitting."
        )
    historical, models, indicators = st.tabs(
        ["Historical Analysis", "AI Forecast Models", "Technical Indicators"],
        key="research_tab",
        on_change="rerun",
    )
    if historical.open:
        with historical:
            render_history(data, symbol, dark, currency)
    if models.open:
        with models:
            render_models(data, horizon, symbol, metadata, dark, currency)
    if indicators.open:
        with indicators:
            render_indicators(data, dark, currency)
    st.caption(
        "Data: Yahoo Finance via yfinance · Research estimates; no guaranteed forecasting accuracy."
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
