# modules/ai/long_hedge_model_sh.py
# Streamlit module: SPY Long + SH Hedge Overlay
# Enhancements included:
# 1) Price-damage hedge trigger (below MA50 + rolling drawdown)
# 2) Dynamic hedge selection (SH vs cash) based on SH trend quality
# 3) Volatility-parity hedge sizing (risk targeting)
# 4) Correlation gate (SPY vs SH) to scale hedge if hedge isn’t behaving inversely
# 5) Hedge min-hold (anti-churn), enforced only when hedge >= hedge_hold_min_level
# 6) Walk-forward validation + parameter stability report

import numpy as np
import pandas as pd
import streamlit as st
def _plot_series(y: pd.Series, title: str):
    st.subheader(title)
    dfp = y.dropna().to_frame(name=title)
    st.line_chart(dfp)


# =============================
# Utilities
# =============================
def zscore(series: pd.Series, lookback: int = 252) -> pd.Series:
    mu = series.rolling(lookback).mean()
    sd = series.rolling(lookback).std(ddof=0)
    return (series - mu) / (sd + 1e-12)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()

def slope(series: pd.Series, window: int = 200) -> pd.Series:
    x = np.arange(window)

    def _s(y):
        if np.any(np.isnan(y)):
            return np.nan
        b = np.cov(x, y, bias=True)[0, 1] / (np.var(x) + 1e-12)
        return b

    return series.rolling(window).apply(_s, raw=True)

def percentile_rank(series: pd.Series, window: int = 252) -> pd.Series:
    def _pr(x):
        s = pd.Series(x)
        return float(s.rank(pct=True).iloc[-1])
    return series.rolling(window).apply(_pr, raw=False)

def rolling_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window).std(ddof=0)

def rolling_corr(a: pd.Series, b: pd.Series, window: int = 20) -> pd.Series:
    return a.rolling(window).corr(b)

def safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default

# =============================
# Hedge sizing helpers
# =============================
def vol_parity_hedge_exposure(desired_hedge: float,
                              vol_spy: float,
                              vol_sh: float,
                              hedge_risk_target: float,
                              min_vol: float = 0.0005,
                              max_mult: float = 2.0) -> float:
    """
    Converts desired hedge *exposure* into a volatility-parity adjusted exposure.
    hedge_risk_target in [0,1]:
      0 -> no adjustment
      1 -> full adjustment (scale by vol_spy / vol_sh, clamped)
    """
    desired_hedge = max(0.0, float(desired_hedge))
    hedge_risk_target = float(np.clip(hedge_risk_target, 0.0, 1.0))

    vol_spy = max(min_vol, float(vol_spy))
    vol_sh = max(min_vol, float(vol_sh))

    mult = vol_spy / vol_sh
    mult = float(np.clip(mult, 1.0 / max_mult, max_mult))

    adjusted = desired_hedge * ((1.0 - hedge_risk_target) + hedge_risk_target * mult)
    return float(max(0.0, adjusted))

def correlation_gate_multiplier(corr_value: float,
                                corr_full_on: float = -0.30,
                                corr_off: float = 0.00,
                                nan_default: float = 1.0) -> float:
    """
    Scales hedge exposure in [0,1] based on SPY~SH correlation.
    - corr <= corr_full_on => 1
    - corr >= corr_off     => 0
    - linear in between
    """
    if np.isnan(corr_value):
        return float(np.clip(nan_default, 0.0, 1.0))

    # Ensure ordering
    if corr_full_on > corr_off:
        corr_full_on, corr_off = corr_off, corr_full_on

    if corr_value <= corr_full_on:
        return 1.0
    if corr_value >= corr_off:
        return 0.0

    return float((corr_off - corr_value) / (corr_off - corr_full_on + 1e-12))

# =============================
# Regime detection
# =============================
def detect_regime(df: pd.DataFrame) -> pd.Series:
    """
    Regime based on:
    - 200d slope of SPY
    - 100d slope of HYG/SPY (credit proxy)
    - VIX percentile
    """
    spy = df["SPY_Close"]
    hyg = df["HYG_Close"]
    vix = df["VIX_Close"]

    sl_200 = slope(spy, 200)
    credit_ratio = hyg / (spy + 1e-12)
    cr_sl_100 = slope(credit_ratio, 100)
    vix_pct = percentile_rank(vix, 252)

    risk_on = (sl_200 > 0) & (cr_sl_100 >= 0) & (vix_pct < 0.75)
    risk_off = (sl_200 < 0) & (cr_sl_100 < 0) & (vix_pct > 0.60)

    regime = pd.Series(index=df.index, dtype="object")
    regime[risk_on] = "risk_on"
    regime[risk_off] = "risk_off"
    regime[regime.isna()] = "transition"
    return regime

def regime_params(regime: str) -> dict:
    # caps and risk controls by regime
    if regime == "risk_on":
        return dict(long_cap=1.00, hedge_cap=0.15, atr_mult=1.2, time_stop=15, cooldown=2)
    if regime == "risk_off":
        return dict(long_cap=0.35, hedge_cap=0.65, atr_mult=0.6, time_stop=7, cooldown=3)
    return dict(long_cap=0.60, hedge_cap=0.35, atr_mult=0.8, time_stop=10, cooldown=3)

# =============================
# Breadth decay + probability engine
# =============================
def breadth_decay_weight(divergence_flag: pd.Series,
                         vol_pct: pd.Series,
                         max_days: int = 20) -> pd.Series:
    w = pd.Series(0.0, index=divergence_flag.index)
    active = False
    start_i = None

    for i in range(len(divergence_flag)):
        flag = bool(divergence_flag.iloc[i])

        if flag and not active:
            active = True
            start_i = i
        if not flag and active:
            active = False
            start_i = None

        if active:
            age = i - start_i
            accel = 0.7 + 0.6 * float(vol_pct.iloc[i])  # ~0.7..1.3
            eff_max = max(5.0, max_days / accel)
            w.iloc[i] = max(0.0, 1.0 - (age / eff_max))
        else:
            w.iloc[i] = 0.0

    return w

def build_probability_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["vix_pct"] = percentile_rank(df["VIX_Close"], 252)
    out["putcall_z"] = zscore(df["PutCall"], 252)
    out["newlows_z"] = -zscore(df["NewLows"], 252)

    spy = df["SPY_Close"]
    new_lows = df["NewLows"]

    price_ll = spy < spy.rolling(10).min().shift(1)
    newlows_not_confirm = new_lows < new_lows.rolling(10).max().shift(1)
    out["breadth_div_flag"] = (price_ll & newlows_not_confirm)

    out["breadth_decay"] = breadth_decay_weight(out["breadth_div_flag"], out["vix_pct"], max_days=20)
    out["breadth_score"] = out["newlows_z"] * (0.5 + 0.5 * out["breadth_decay"])

    # Total probability score (probability shaper, not a timing signal)
    out["prob_score"] = 0.7 * out["breadth_score"] + 0.6 * out["putcall_z"]
    return out

def price_confirmation(spy_close: pd.Series) -> pd.Series:
    ma20 = spy_close.rolling(20).mean()
    return spy_close > ma20

# =============================
# Enhancement: price damage
# =============================
def price_damage_flags(spy_close: pd.Series,
                       ma_len: int = 50,
                       dd_window: int = 63,
                       dd_trigger: float = -0.06) -> pd.DataFrame:
    out = pd.DataFrame(index=spy_close.index)

    ma = spy_close.rolling(ma_len).mean()
    out["below_ma50"] = spy_close < ma

    roll_peak = spy_close.rolling(dd_window).max()
    roll_dd = spy_close / (roll_peak + 1e-12) - 1.0
    out["roll_dd"] = roll_dd
    out["dd_damage"] = roll_dd <= dd_trigger
    return out

# =============================
# Enhancement: dynamic hedge selection (SH vs cash)
# =============================
def hedge_quality(sh_close: pd.Series, ma_len: int = 20, slope_len: int = 60) -> pd.Series:
    ma = sh_close.rolling(ma_len).mean()
    sl = slope(sh_close, slope_len)
    ok = (sh_close > ma) & (sl > 0)
    return ok.fillna(False)

# =============================
# Allocation logic
# =============================
def desired_allocations(regime: str,
                        prob: float,
                        confirm: bool,
                        entry_th: float,
                        hedge_th: float,
                        damage_below_ma50: bool,
                        damage_dd: bool,
                        damage_weight: float) -> tuple[float, float]:
    """
    Long:
      - requires confirm and prob > entry_th
      - ramps position with strength
    Hedge:
      - defensive overlay driven by regime + (prob weak OR confirm fails OR price damage)
    """
    rp = regime_params(regime)
    long_cap = rp["long_cap"]
    hedge_cap = rp["hedge_cap"]

    # Long desire
    if confirm and prob > entry_th:
        strength = min(1.0, max(0.0, (prob - entry_th) / 2.0))
        desired_long = long_cap * (0.40 + 0.60 * strength)
    else:
        desired_long = 0.0

    # Price damage scalar
    dmg_score = 0.0
    if damage_below_ma50:
        dmg_score += 1.0
    if damage_dd:
        dmg_score += 1.0
    damage_scalar = min(1.0, (dmg_score / 2.0) * float(np.clip(damage_weight, 0.0, 1.0)))

    desired_hedge = 0.0
    if regime == "risk_off":
        if (not confirm) or (prob < hedge_th) or (damage_scalar > 0):
            weakness = min(1.0, max(0.0, (hedge_th - prob) / 2.0))
            desired_hedge = hedge_cap * (0.50 + 0.50 * max(weakness, damage_scalar))
        else:
            desired_hedge = hedge_cap * 0.35
    elif regime == "transition":
        if (not confirm) or (prob < hedge_th) or (damage_scalar > 0):
            weakness = min(1.0, max(0.0, (hedge_th - prob) / 2.0))
            desired_hedge = hedge_cap * (0.30 + 0.70 * max(weakness, damage_scalar))
    else:  # risk_on
        if ((not confirm) and (prob < hedge_th)) or (damage_scalar > 0):
            desired_hedge = min(hedge_cap, 0.10 + 0.10 * damage_scalar)

    # Gross cap safety
    if desired_long + desired_hedge > 1.0:
        s = 1.0 / (desired_long + desired_hedge)
        desired_long *= s
        desired_hedge *= s

    return float(desired_long), float(desired_hedge)

# =============================
# Backtest engine
# =============================
def performance_metrics(bt: pd.DataFrame, annualization: int = 252) -> dict:
    r = bt["returns"]
    eq = bt["equity"]
    if len(eq) < 2:
        return {}
    cagr = (eq.iloc[-1] ** (annualization / len(eq))) - 1.0
    vol = r.std() * np.sqrt(annualization)
    sharpe = (r.mean() * annualization) / (r.std() * np.sqrt(annualization) + 1e-12)
    dd = (eq / eq.cummax() - 1.0).min()
    return {
        "CAGR": float(cagr),
        "Vol": float(vol),
        "Sharpe": float(sharpe),
        "MaxDD": float(dd),
        "FinalEquity": float(eq.iloc[-1]),
    }

def backtest_long_hedge_sh(df: pd.DataFrame,
                           entry_th: float = 1.0,
                           hedge_th: float = 0.0,
                           fee_bps: float = 1.0,
                           slippage_bps: float = 1.0,
                           hedge_step: float = 0.20,
                           dd_trigger: float = -0.06,
                           damage_weight: float = 1.0,
                           dynamic_hedge_enabled: bool = True,
                           hedge_risk_target: float = 0.75,
                           corr_full_on: float = -0.30,
                           corr_off: float = 0.00,
                           corr_gate_enabled: bool = True,
                           hedge_min_hold_days: int = 3,
                           hedge_hold_min_level: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = df.copy().sort_index()

    # Features
    regime = detect_regime(df)
    prob = build_probability_engine(df)
    confirm = price_confirmation(df["SPY_Close"])
    df["SPY_ATR"] = atr(df["SPY_High"], df["SPY_Low"], df["SPY_Close"], 14)

    dmg = price_damage_flags(df["SPY_Close"], ma_len=50, dd_window=63, dd_trigger=dd_trigger)
    hq = hedge_quality(df["SH_Close"], ma_len=20, slope_len=60)

    # Returns/vol/corr for parity + correlation gate
    df["SPY_ret"] = df["SPY_Close"].pct_change().fillna(0.0)
    df["SH_ret"] = df["SH_Close"].pct_change().fillna(0.0)
    df["SPY_vol"] = rolling_vol(df["SPY_ret"], window=20)
    df["SH_vol"] = rolling_vol(df["SH_ret"], window=20)
    df["corr_spy_sh"] = rolling_corr(df["SPY_ret"], df["SH_ret"], window=20)

    # State
    long_pos = 0.0
    hedge_pos = 0.0

    long_entry_day = None
    long_stop = np.nan
    cooldown = 0

    hedge_hold_timer = 0

    equity = 1.0
    eq, lp, hp = [], [], []
    trades = []

    gate_hist = []
    hedge_hold_hist = []
    hedge_meaningful_hist = []

    def cost_hit():
        nonlocal equity
        cost = (fee_bps + slippage_bps) / 10000.0
        equity *= (1.0 - cost)

    for i in range(len(df)):
        dt = df.index[i]
        spy = float(df["SPY_Close"].iloc[i])
        sh = float(df["SH_Close"].iloc[i])
        reg = str(regime.iloc[i])

        pscore = safe_float(prob["prob_score"].iloc[i], 0.0)
        ok = bool(confirm.iloc[i])

        below_ma50 = bool(dmg["below_ma50"].iloc[i]) if "below_ma50" in dmg.columns else False
        dd_damage = bool(dmg["dd_damage"].iloc[i]) if "dd_damage" in dmg.columns else False

        hedge_ok = bool(hq.iloc[i]) if dynamic_hedge_enabled else True
        params = regime_params(reg)

        # MTM PnL
        if i > 0:
            spy_prev = float(df["SPY_Close"].iloc[i - 1])
            sh_prev = float(df["SH_Close"].iloc[i - 1])
            r_spy = (spy / spy_prev) - 1.0
            r_sh = (sh / sh_prev) - 1.0
            equity *= (1.0 + long_pos * r_spy + hedge_pos * r_sh)

        # cooldown decrement
        if cooldown > 0:
            cooldown -= 1

        # hedge hold decrement
        if hedge_hold_timer > 0:
            hedge_hold_timer -= 1

        # Long risk controls
        if long_pos > 0:
            if spy <= long_stop:
                cost_hit()
                trades.append((dt, "LONG_STOP_EXIT", long_pos, spy, equity))
                long_pos = 0.0
                long_entry_day = None
                long_stop = np.nan
                cooldown = params["cooldown"]

            if long_pos > 0 and long_entry_day is not None:
                age_days = (dt - long_entry_day).days
                if age_days >= params["time_stop"]:
                    cost_hit()
                    trades.append((dt, "LONG_TIME_EXIT", long_pos, spy, equity))
                    long_pos = 0.0
                    long_entry_day = None
                    long_stop = np.nan
                    cooldown = max(1, params["cooldown"] - 1)

        # Desired allocations (base)
        des_long, des_hedge = desired_allocations(
            regime=reg,
            prob=pscore,
            confirm=ok,
            entry_th=float(entry_th),
            hedge_th=float(hedge_th),
            damage_below_ma50=below_ma50,
            damage_dd=dd_damage,
            damage_weight=float(damage_weight),
        )

        # Vol-parity sizing for hedge exposure
        vol_spy = safe_float(df["SPY_vol"].iloc[i], 0.0)
        vol_sh = safe_float(df["SH_vol"].iloc[i], 0.0)
        des_hedge = vol_parity_hedge_exposure(
            desired_hedge=des_hedge,
            vol_spy=vol_spy,
            vol_sh=vol_sh,
            hedge_risk_target=float(hedge_risk_target),
        )

        # Correlation gate
        gate_val = 1.0
        if corr_gate_enabled:
            corr_val = safe_float(df["corr_spy_sh"].iloc[i], np.nan)
            gate_val = correlation_gate_multiplier(
                corr_value=corr_val,
                corr_full_on=float(corr_full_on),
                corr_off=float(corr_off),
                nan_default=1.0,
            )
            des_hedge *= gate_val

        # Dynamic hedge selection (SH vs cash)
        if dynamic_hedge_enabled and (not hedge_ok):
            des_hedge = 0.0

        # Staged long entries on follow-through only
        follow = (i > 0 and spy > float(df["SPY_Close"].iloc[i - 1]))
        if cooldown == 0 and des_long > 0:
            if follow and long_pos < des_long:
                add = min(0.25 * des_long, des_long - long_pos)
                if add > 0:
                    cost_hit()
                    long_pos += add
                    trades.append((dt, "LONG_ADD", add, spy, equity))
                    if long_entry_day is None:
                        long_entry_day = dt
                    atr_i = safe_float(df["SPY_ATR"].iloc[i], 0.0)
                    long_stop = spy - params["atr_mult"] * atr_i

        # Exit long if signal off (clean)
        if des_long == 0.0 and long_pos > 0 and cooldown == 0:
            cost_hit()
            trades.append((dt, "LONG_EXIT_SIGNAL", long_pos, spy, equity))
            long_pos = 0.0
            long_entry_day = None
            long_stop = np.nan
            cooldown = max(1, params["cooldown"] - 1)

        # =========================
        # Hedge smoothing + min-hold (only when hedge is meaningful)
        # =========================
        hedge_hold_min_level = float(np.clip(hedge_hold_min_level, 0.0, 1.0))
        meaningful = (hedge_pos >= hedge_hold_min_level)

        if des_hedge > hedge_pos + hedge_step:
            # Increase hedge (always allowed)
            cost_hit()
            new_hedge = min(des_hedge, hedge_pos + hedge_step)
            inc = new_hedge - hedge_pos
            hedge_pos = new_hedge
            trades.append((dt, "HEDGE_INCREASE", inc, sh, equity))

            # Start/refresh hold only if meaningful
            if hedge_pos >= hedge_hold_min_level and hedge_min_hold_days > 0:
                hedge_hold_timer = int(hedge_min_hold_days)

        elif des_hedge < hedge_pos - hedge_step:
            # Decrease hedge (blocked if hold active AND meaningful)
            if hedge_hold_timer > 0 and meaningful:
                pass
            else:
                cost_hit()
                new_hedge = max(des_hedge, hedge_pos - hedge_step)
                dec = hedge_pos - new_hedge
                hedge_pos = new_hedge
                trades.append((dt, "HEDGE_DECREASE", dec, sh, equity))
        else:
            # Small changes: respect hold only if decreasing meaningful hedge
            if des_hedge < hedge_pos and hedge_hold_timer > 0 and meaningful:
                pass
            else:
                hedge_pos = des_hedge

        # Gross cap safety
        if long_pos + hedge_pos > 1.0:
            s = 1.0 / (long_pos + hedge_pos)
            long_pos *= s
            hedge_pos *= s

        # Diagnostics
        gate_hist.append(gate_val)
        hedge_hold_hist.append(hedge_hold_timer)
        hedge_meaningful_hist.append(1 if hedge_pos >= hedge_hold_min_level else 0)

        eq.append(equity)
        lp.append(long_pos)
        hp.append(hedge_pos)

    bt = pd.DataFrame(index=df.index)
    bt["equity"] = eq
    bt["returns"] = bt["equity"].pct_change().fillna(0.0)
    bt["drawdown"] = bt["equity"] / bt["equity"].cummax() - 1.0

    bt["long_pos"] = lp
    bt["hedge_pos"] = hp
    bt["gross"] = (bt["long_pos"] + bt["hedge_pos"]).clip(upper=1.0)

    bt["regime"] = regime
    bt["prob_score"] = prob["prob_score"]
    bt["vix_pct"] = prob["vix_pct"]
    bt["putcall_z"] = prob["putcall_z"]
    bt["breadth_score"] = prob["breadth_score"]
    bt["breadth_decay"] = prob["breadth_decay"]

    bt["below_ma50"] = dmg["below_ma50"]
    bt["roll_dd"] = dmg["roll_dd"]
    bt["dd_damage"] = dmg["dd_damage"]
    bt["hedge_quality_ok"] = hq

    bt["SPY_vol"] = df["SPY_vol"]
    bt["SH_vol"] = df["SH_vol"]
    bt["corr_spy_sh"] = df["corr_spy_sh"]
    bt["hedge_corr_gate"] = gate_hist
    bt["hedge_hold_timer"] = hedge_hold_hist
    bt["hedge_meaningful"] = hedge_meaningful_hist

    trade_log = pd.DataFrame(trades, columns=["date", "action", "size", "price", "equity"]).set_index("date")
    stats = performance_metrics(bt)
    return bt, trade_log, stats

# =============================
# Walk-forward validation
# =============================
def walk_forward_validation(df: pd.DataFrame,
                            param_grid: list[dict],
                            train_years: float = 3.0,
                            test_years: float = 1.0,
                            step_months: int = 6,
                            fee_bps: float = 1.0,
                            slippage_bps: float = 1.0,
                            hedge_step: float = 0.20,
                            dd_trigger: float = -0.06,
                            damage_weight: float = 1.0,
                            dynamic_hedge_enabled: bool = True,
                            hedge_risk_target: float = 0.75,
                            corr_full_on: float = -0.30,
                            corr_off: float = 0.00,
                            corr_gate_enabled: bool = True,
                            hedge_min_hold_days: int = 3,
                            hedge_hold_min_level: float = 0.20) -> pd.DataFrame:
    df = df.copy().sort_index()
    if len(df) < 400:
        return pd.DataFrame()

    train_days = int(252 * train_years)
    test_days = int(252 * test_years)
    step_days = int(21 * step_months)

    rows = []
    start = 0

    while True:
        train_start = start
        train_end = train_start + train_days
        test_end = train_end + test_days
        if test_end >= len(df):
            break

        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[train_end:test_end]

        best = None
        best_sharpe = -1e9
        best_dd = -1e9

        for pg in param_grid:
            bt_tr, _, st_tr = backtest_long_hedge_sh(
                train_df,
                entry_th=float(pg["entry_th"]),
                hedge_th=float(pg["hedge_th"]),
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                hedge_step=hedge_step,
                dd_trigger=dd_trigger,
                damage_weight=damage_weight,
                dynamic_hedge_enabled=dynamic_hedge_enabled,
                hedge_risk_target=hedge_risk_target,
                corr_full_on=corr_full_on,
                corr_off=corr_off,
                corr_gate_enabled=corr_gate_enabled,
                hedge_min_hold_days=hedge_min_hold_days,
                hedge_hold_min_level=hedge_hold_min_level,
            )
            sharpe = st_tr.get("Sharpe", np.nan)
            maxdd = st_tr.get("MaxDD", np.nan)
            if np.isnan(sharpe) or np.isnan(maxdd):
                continue

            # primary: sharpe; tie-break: maxdd (less negative is better => larger value)
            if (sharpe > best_sharpe) or (np.isclose(sharpe, best_sharpe) and maxdd > best_dd):
                best_sharpe = sharpe
                best_dd = maxdd
                best = pg

        if best is None:
            start += step_days
            continue

        bt_te, _, st_te = backtest_long_hedge_sh(
            test_df,
            entry_th=float(best["entry_th"]),
            hedge_th=float(best["hedge_th"]),
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            hedge_step=hedge_step,
            dd_trigger=dd_trigger,
            damage_weight=damage_weight,
            dynamic_hedge_enabled=dynamic_hedge_enabled,
            hedge_risk_target=hedge_risk_target,
            corr_full_on=corr_full_on,
            corr_off=corr_off,
            corr_gate_enabled=corr_gate_enabled,
            hedge_min_hold_days=hedge_min_hold_days,
            hedge_hold_min_level=hedge_hold_min_level,
        )

        rows.append({
            "train_start": train_df.index[0],
            "train_end": train_df.index[-1],
            "test_start": test_df.index[0],
            "test_end": test_df.index[-1],
            "selected_entry_th": float(best["entry_th"]),
            "selected_hedge_th": float(best["hedge_th"]),
            "test_CAGR": st_te.get("CAGR", np.nan),
            "test_Sharpe": st_te.get("Sharpe", np.nan),
            "test_MaxDD": st_te.get("MaxDD", np.nan),
            "test_FinalEquity": st_te.get("FinalEquity", np.nan),
        })

        start += step_days

    return pd.DataFrame(rows)

# =============================
# Plot helpers
# =============================
def _plot_series(y: pd.Series, title: str):
    fig = plt.figure()
    plt.plot(y.index, y.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.grid(True)
    st.pyplot(fig)

# =============================
# Data loading/validation
# =============================
def _validate_columns(df: pd.DataFrame) -> list[str]:
    required = [
        "SPY_Open", "SPY_High", "SPY_Low", "SPY_Close",
        "SH_Close", "VIX_Close", "HYG_Close", "PutCall", "NewLows"
    ]
    return [c for c in required if c not in df.columns]

def _load_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    else:
        df.index = pd.to_datetime(df.index)
    return df.sort_index()

# =============================
# Streamlit UI
# =============================
def run():
    st.title("SPY Long + SH Hedge Overlay (Enhanced + Walk-Forward)")

    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload CSV (daily)", type=["csv"])

        st.header("Core Thresholds")
        entry_th = st.slider("Long Entry Threshold (prob_score)", -1.0, 3.0, 1.0, 0.1)
        hedge_th = st.slider("Hedge Threshold (prob_score)", -3.0, 2.0, 0.0, 0.1)

        st.header("Costs & Smoothing")
        fee_bps = st.slider("Fees (bps per trade action)", 0.0, 10.0, 1.0, 0.5)
        slippage_bps = st.slider("Slippage (bps per trade action)", 0.0, 20.0, 1.0, 0.5)
        hedge_step = st.slider("Max Hedge Change per Day", 0.05, 0.50, 0.20, 0.05)

        st.header("Price Damage Triggers")
        dd_trigger = st.slider("Rolling DD Trigger (63d) for damage", -0.20, -0.01, -0.06, 0.01)
        damage_weight = st.slider("Damage Weight (0..1)", 0.0, 1.0, 1.0, 0.1)

        st.header("Dynamic Hedge Selection")
        dynamic_hedge_enabled = st.toggle("Use SH only when hedge quality is good (else cash)", value=True)

        st.header("Hedge Risk Sizing (Vol-Parity)")
        hedge_risk_target = st.slider("Hedge Risk Target (0=no parity, 1=full parity)", 0.0, 1.0, 0.75, 0.05)

        st.header("Correlation Gate (SPY vs SH)")
        corr_gate_enabled = st.toggle("Enable correlation gate", value=True)
        corr_full_on = st.slider("Full hedge if corr <= ", -1.0, 0.0, -0.30, 0.05)
        corr_off = st.slider("No hedge if corr >= ", -0.50, 0.50, 0.00, 0.05)

        st.header("Anti-Churn: Hedge Min-Hold")
        hedge_min_hold_days = st.slider("Min hold days after hedge increases", 0, 10, 3, 1)
        hedge_hold_min_level = st.slider("Only enforce min-hold when hedge >= ", 0.0, 1.0, 0.20, 0.05)

        st.header("Walk-Forward (Stability)")
        wf_enable = st.toggle("Run walk-forward validation", value=False)
        train_years = st.slider("Train window (years)", 1.0, 6.0, 3.0, 0.5)
        test_years = st.slider("Test window (years)", 0.5, 3.0, 1.0, 0.5)
        step_months = st.slider("Step size (months)", 1, 12, 6, 1)

        st.caption("Tip: raise entry_th to reduce whipsaw; enable dynamic hedge + corr gate to avoid bad hedges; min-hold reduces churn.")

    if uploaded is None:
        st.info("Upload a CSV with required columns to run the model.")
        st.stop()

    df = _load_csv(uploaded)
    missing = _validate_columns(df)
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    bt, trades, stats = backtest_long_hedge_sh(
        df=df,
        entry_th=float(entry_th),
        hedge_th=float(hedge_th),
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
        hedge_step=float(hedge_step),
        dd_trigger=float(dd_trigger),
        damage_weight=float(damage_weight),
        dynamic_hedge_enabled=bool(dynamic_hedge_enabled),
        hedge_risk_target=float(hedge_risk_target),
        corr_full_on=float(corr_full_on),
        corr_off=float(corr_off),
        corr_gate_enabled=bool(corr_gate_enabled),
        hedge_min_hold_days=int(hedge_min_hold_days),
        hedge_hold_min_level=float(hedge_hold_min_level),
    )

    # Headline stats
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"{stats.get('CAGR', 0.0)*100:.2f}%")
    c2.metric("Vol", f"{stats.get('Vol', 0.0)*100:.2f}%")
    c3.metric("Sharpe", f"{stats.get('Sharpe', 0.0):.2f}")
    c4.metric("Max DD", f"{stats.get('MaxDD', 0.0)*100:.2f}%")
    c5.metric("Final Equity", f"{stats.get('FinalEquity', 1.0):.2f}")

    tabs = st.tabs([
        "Regime",
        "Probability Engine",
        "Positions & Risk",
        "Damage/Hedge Diagnostics",
        "Backtest & Trades",
        "Walk-Forward (Stability)"
    ])

    with tabs[0]:
        st.subheader("Regime Monitor")
        st.write("Latest regime:", bt["regime"].iloc[-1])
        _plot_series(bt["vix_pct"], "VIX Percentile (Rolling)")
        st.dataframe(bt[["regime"]].tail(60))

    with tabs[1]:
        st.subheader("Probability Engine")
        _plot_series(bt["prob_score"], "Total Probability Score")
        _plot_series(bt["putcall_z"], "Put/Call Z-Score (Sentiment)")
        _plot_series(bt["breadth_score"], "Breadth Score (New Lows Z with Decay)")
        _plot_series(bt["breadth_decay"], "Breadth Divergence Decay Weight")
        st.dataframe(bt[["prob_score", "putcall_z", "breadth_score", "breadth_decay"]].tail(60))

    with tabs[2]:
        st.subheader("Positions & Risk")
        _plot_series(bt["long_pos"], "Long Exposure (SPY)")
        _plot_series(bt["hedge_pos"], "Hedge Exposure (SH)")
        _plot_series(bt["gross"], "Gross Exposure (Long + Hedge)")
        _plot_series(bt["drawdown"], "Drawdown")
        st.dataframe(bt[["long_pos", "hedge_pos", "gross", "regime", "prob_score"]].tail(30))

    with tabs[3]:
        st.subheader("Damage + Hedge Diagnostics")
        _plot_series(bt["roll_dd"], "Rolling Drawdown vs 63d peak (SPY)")
        _plot_series(bt["below_ma50"].astype(int), "Below 50D MA flag (1=yes)")
        _plot_series(bt["dd_damage"].astype(int), "DD Damage flag (1=yes)")
        _plot_series(bt["hedge_quality_ok"].astype(int), "Hedge Quality OK (SH above MA + slope>0)")
        _plot_series(bt["corr_spy_sh"], "Rolling Corr (SPY_ret vs SH_ret)")
        _plot_series(bt["hedge_corr_gate"], "Correlation Gate Multiplier (0..1)")
        _plot_series(bt["hedge_hold_timer"], "Hedge Hold Timer (days remaining)")
        _plot_series(bt["hedge_meaningful"], "Hedge Meaningful Flag (>= min level)")
        st.dataframe(
            bt[[
                "below_ma50", "dd_damage", "roll_dd",
                "hedge_quality_ok", "corr_spy_sh", "hedge_corr_gate",
                "hedge_hold_timer", "hedge_pos"
            ]].tail(90)
        )

    with tabs[4]:
        st.subheader("Backtest")
        _plot_series(bt["equity"], "Equity Curve")
        st.subheader("Trade Log (last 300)")
        st.dataframe(trades.tail(300))

        st.download_button(
            "Download Backtest (CSV)",
            data=bt.to_csv().encode("utf-8"),
            file_name="backtest_spy_long_sh_hedge_final.csv",
            mime="text/csv"
        )
        st.download_button(
            "Download Trades (CSV)",
            data=trades.to_csv().encode("utf-8"),
            file_name="trades_spy_long_sh_hedge_final.csv",
            mime="text/csv"
        )

       with tabs[5]:
        st.subheader("Walk-Forward Validation & Parameter Stability")

        if not wf_enable:
            st.info("Enable walk-forward in the sidebar to compute stability report.")
        else:
            # Small grid around current values
            entry_vals = [round(x, 2) for x in np.arange(entry_th - 0.5, entry_th + 0.51, 0.25)]
            hedge_vals = [round(x, 2) for x in np.arange(hedge_th - 0.5, hedge_th + 0.51, 0.25)]
            param_grid = [{"entry_th": e, "hedge_th": h} for e in entry_vals for h in hedge_vals]

            wf = walk_forward_validation(
                df=df,
                param_grid=param_grid,
                train_years=float(train_years),
                test_years=float(test_years),
                step_months=int(step_months),
                fee_bps=float(fee_bps),
                slippage_bps=float(slippage_bps),
                hedge_step=float(hedge_step),
                dd_trigger=float(dd_trigger),
                damage_weight=float(damage_weight),
                dynamic_hedge_enabled=bool(dynamic_hedge_enabled),
                hedge_risk_target=float(hedge_risk_target),
                corr_full_on=float(corr_full_on),
                corr_off=float(corr_off),
                corr_gate_enabled=bool(corr_gate_enabled),
                hedge_min_hold_days=int(hedge_min_hold_days),
                hedge_hold_min_level=float(hedge_hold_min_level),
            )

            if wf.empty:
                st.warning("Not enough data to run walk-forward (need ~400+ daily rows).")
                return

            st.write("Walk-forward windows and selected parameters:")
            st.dataframe(wf, use_container_width=True)

            st.subheader("Parameter stability (selection frequency)")
            freq_entry = wf["selected_entry_th"].value_counts().sort_index()
            freq_hedge = wf["selected_hedge_th"].value_counts().sort_index()

            st.write("Selected entry_th frequency")
            st.bar_chart(freq_entry)

            st.write("Selected hedge_th frequency")
            st.bar_chart(freq_hedge)

            st.subheader("Out-of-sample performance summary (median)")
            oos = {
                "OOS Sharpe (median)": float(wf["test_Sharpe"].median()),
                "OOS MaxDD (median)": float(wf["test_MaxDD"].median()),
                "OOS CAGR (median)": float(wf["test_CAGR"].median()),
                "OOS windows": int(len(wf)),
            }
            st.json(oos)

            st.download_button(
                "Download Walk-Forward Results (CSV)",
                data=wf.to_csv(index=False).encode("utf-8"),
                file_name="walk_forward_stability_report.csv",
                mime="text/csv"
            )
