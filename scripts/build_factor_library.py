"""Build the 80-factor library JSON used by the LLM idea-agent.

Every factor here MUST parse through AlphaGen's ExpressionParser using the
allow-list in `alphagen.config.OPERATORS`. We run a parse-check on every entry
before writing the JSON so syntax errors surface immediately.

Operator allow-list (from alphagen.config.OPERATORS):
    Unary       : Abs, Log
    Binary      : Add, Sub, Mul, Div, Greater, Less
    Rolling     : Ref, Mean, Sum, Std, Var, Max, Min, Med, Mad, Delta, WMA, EMA
    PairRolling : Cov, Corr

Allowed windows : 1d, 5d, 10d, 20d, 40d
Allowed features: $open $high $low $close $volume $vwap
Allowed constants: -30, -10, -5, -2, -1, -0.5, -0.01, 0.01, 0.5, 1, 2, 5, 10, 30

Run:
    python scripts/build_factor_library.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "external" / "alphagen"))

from alphagen.config import OPERATORS
from alphagen.data.parser import ExpressionParser, ExpressionParsingError


def build_parser() -> ExpressionParser:
    return ExpressionParser(
        operators=OPERATORS,
        ignore_case=False,
        time_deltas_need_suffix=True,
        non_positive_time_deltas_allowed=False,
        feature_need_dollar_sign=True,
    )


# ---------------------------------------------------------------------------
# Factor catalog. Each tuple: (id, family, expr, description, source)
# ---------------------------------------------------------------------------
FACTORS: list[tuple[str, str, str, str, str]] = [
    # ====================================================================
    # MOMENTUM & REVERSAL  (15)
    # ====================================================================
    ("m_001", "momentum", "Div(Sub($close,Ref($close,5d)),Ref($close,5d))",
     "5-day cumulative return", "WQ101"),
    ("m_002", "momentum", "Div(Sub($close,Ref($close,10d)),Ref($close,10d))",
     "10-day cumulative return", "WQ101"),
    ("m_003", "momentum", "Div(Sub($close,Ref($close,20d)),Ref($close,20d))",
     "20-day cumulative return", "WQ101"),
    ("m_004", "momentum", "Div(Sub($close,Ref($close,40d)),Ref($close,40d))",
     "40-day cumulative return", "GTJA191"),
    ("m_005", "momentum", "Mean(Delta($close,5d),10d)",
     "10-day average of 5-day price change", "GTJA191"),
    ("m_006", "momentum", "Div(Delta($close,5d),Std($close,20d))",
     "Volatility-normalized 5-day momentum (Sharpe-like)", "HTSY"),
    ("m_007", "momentum", "Div(Delta($close,10d),Std($close,20d))",
     "Volatility-normalized 10-day momentum", "HTSY"),
    ("m_008", "reversal", "Mul(-1.0,Delta($close,1d))",
     "1-day short-term reversal", "WQ101"),
    ("m_009", "reversal", "Mul(-1.0,Delta($close,5d))",
     "5-day short-term reversal (Jegadeesh-Titman style)", "WQ101"),
    ("m_010", "momentum", "Div(Sub($close,Mean($close,10d)),Mean($close,10d))",
     "Distance from 10-day moving average", "GTJA191"),
    ("m_011", "momentum", "Div(Sub($close,Max($close,20d)),$close)",
     "Distance to 20-day high (negative when below high)", "GTJA191"),
    ("m_012", "momentum", "Div(Sub($close,Min($close,20d)),$close)",
     "Distance from 20-day low", "GTJA191"),
    ("m_013", "momentum", "Div(Mean($close,5d),Mean($close,20d))",
     "5-day MA / 20-day MA (golden-cross proxy)", "HTSY"),
    ("m_014", "momentum", "Sub(EMA($close,5d),EMA($close,20d))",
     "MACD-like EMA(5)-EMA(20) momentum", "WQ101"),
    ("m_015", "momentum", "Div(Sub($close,Mean($close,20d)),Std($close,20d))",
     "20-day z-score of close (Bollinger-style)", "SWS"),

    # ====================================================================
    # VOLUME & PRICE-VOLUME  (15)
    # ====================================================================
    ("v_001", "volume", "Div($volume,Mean($volume,5d))",
     "5-day relative volume ratio", "GTJA191"),
    ("v_002", "volume", "Div($volume,Mean($volume,20d))",
     "20-day relative volume ratio", "GTJA191"),
    ("v_003", "volume", "Log(Add($volume,1.0))",
     "Log-volume (always positive, well-behaved)", "WQ101"),
    ("v_004", "volume", "Delta($volume,5d)",
     "5-day volume change", "GTJA191"),
    ("v_005", "volume", "Std($volume,20d)",
     "20-day volume volatility", "WQ101"),
    ("v_006", "volume", "Sub($volume,Mean($volume,20d))",
     "Abnormal volume vs 20-day average", "GTJA191"),
    ("v_007", "vol_corr", "Corr($close,$volume,10d)",
     "10-day close-volume rolling correlation", "WQ101"),
    ("v_008", "vol_corr", "Corr($close,$volume,20d)",
     "20-day close-volume rolling correlation", "WQ101"),
    ("v_009", "vol_corr", "Mul(-1.0,Corr($close,$volume,20d))",
     "Negative close-volume correlation (volume-divergence reversal)", "HTSY"),
    ("v_010", "vol_corr", "Corr(Delta($close,1d),$volume,20d)",
     "20-day return-volume correlation", "WQ101"),
    ("v_011", "vol_corr", "Cov($close,$volume,20d)",
     "20-day price-volume covariance", "GTJA191"),
    ("v_012", "volume", "Mul(Div($volume,Mean($volume,5d)),Delta($close,1d))",
     "Volume-confirmed daily return", "HTSY"),
    ("v_013", "volume", "Div(Mean($volume,5d),Mean($volume,20d))",
     "Short-term vs long-term volume regime", "SWS"),
    ("v_014", "vol_corr", "Corr($high,$volume,10d)",
     "High-price vs volume correlation (buy-pressure proxy)", "GTJA191"),
    ("v_015", "vol_corr", "Corr($low,$volume,10d)",
     "Low-price vs volume correlation (sell-pressure proxy)", "GTJA191"),

    # ====================================================================
    # VOLATILITY  (10)
    # ====================================================================
    ("vol_001", "vol_regime", "Std($close,5d)",
     "5-day realized volatility of close", "WQ101"),
    ("vol_002", "vol_regime", "Std($close,20d)",
     "20-day realized volatility", "WQ101"),
    ("vol_003", "vol_regime", "Div(Std($close,5d),Std($close,20d))",
     "Short vs long vol ratio (vol regime indicator)", "HTSY"),
    ("vol_004", "vol_regime", "Delta(Std($close,20d),5d)",
     "5-day change in 20-day vol", "GTJA191"),
    ("vol_005", "vol_regime", "Var($close,20d)",
     "20-day return variance", "WQ101"),
    ("vol_006", "vol_regime", "Std(Delta($close,1d),20d)",
     "20-day std of daily returns", "GTJA191"),
    ("vol_007", "vol_regime", "Std(Sub($high,$low),20d)",
     "Volatility of daily high-low range", "SWS"),
    ("vol_008", "vol_regime", "Div(Std($close,5d),Mean(Std($close,5d),20d))",
     "Vol regime: current vs trailing-average vol", "HTSY"),
    ("vol_009", "vol_regime", "Mul(-1.0,Std($close,5d))",
     "Negative vol (low-vol bias)", "SWS"),
    ("vol_010", "vol_regime", "Std($vwap,20d)",
     "20-day VWAP volatility", "GTJA191"),

    # ====================================================================
    # RANGE & 振幅  (10)
    # ====================================================================
    ("r_001", "range", "Sub($high,$low)",
     "Daily high-low range", "GTJA191"),
    ("r_002", "range", "Div(Sub($high,$low),$close)",
     "Daily range normalized by close", "GTJA191"),
    ("r_003", "range", "Sub(Max($close,5d),Min($close,5d))",
     "5-day price range (max-min)", "WQ101"),
    ("r_004", "range", "Div(Sub(Max($close,20d),Min($close,20d)),$close)",
     "20-day range / close", "HTSY"),
    ("r_005", "range", "Div(Sub($close,Min($low,20d)),Sub(Max($high,20d),Min($low,20d)))",
     "Position within 20-day high-low channel (Williams %R-like)", "HTSY"),
    ("r_006", "range", "Div(Abs(Sub($close,$open)),Sub($high,$low))",
     "Body-to-range ratio (decisiveness)", "SWS"),
    ("r_007", "range", "Div(Sub($high,$low),Mean(Sub($high,$low),20d))",
     "Range expansion vs 20-day avg range", "GTJA191"),
    ("r_008", "range", "Mul(-1.0,Std(Sub($high,$low),5d))",
     "Range compression proxy (NR7-like)", "WQ101"),
    ("r_009", "range", "Mean(Sub($high,$low),20d)",
     "20-day Average True Range proxy", "WQ101"),
    ("r_010", "range", "Div(Sub($high,Greater($open,$close)),Sub($high,$low))",
     "Upper-shadow ratio (selling pressure proxy)", "SWS"),

    # ====================================================================
    # VWAP & MICROSTRUCTURE  (10)
    # ====================================================================
    ("vwp_001", "vwap_dev", "Sub($close,$vwap)",
     "Close minus VWAP (aggressive flow direction)", "WQ101"),
    ("vwp_002", "vwap_dev", "Div(Sub($close,$vwap),$vwap)",
     "Close-VWAP normalized (relative deviation)", "WQ101"),
    ("vwp_003", "vwap_dev", "Delta($vwap,5d)",
     "5-day VWAP change", "GTJA191"),
    ("vwp_004", "vwap_dev", "Div($vwap,Mean($vwap,20d))",
     "VWAP relative to 20-day VWAP MA", "HTSY"),
    ("vwp_005", "vwap_dev", "Mean(Sub($close,$vwap),5d)",
     "5-day mean of close-VWAP gap", "SWS"),
    ("vwp_006", "vwap_dev", "Sub($open,$vwap)",
     "Open minus VWAP (opening pressure)", "WQ101"),
    ("vwp_007", "vwap_dev", "Sub($vwap,Mean($vwap,5d))",
     "VWAP deviation from 5-day MA", "GTJA191"),
    ("vwp_008", "vwap_dev", "Mul(-1.0,Div(Sub($close,$vwap),$vwap))",
     "Reversal of close-VWAP deviation", "HTSY"),
    ("vwp_009", "vwap_dev", "Corr($close,$vwap,10d)",
     "10-day close-VWAP correlation", "WQ101"),
    ("vwp_010", "vwap_dev", "Mul(Sub($close,$vwap),$volume)",
     "Volume-weighted close-VWAP gap (dollar flow)", "GTJA191"),

    # ====================================================================
    # OPEN-CLOSE RELATIONSHIPS  (10)
    # ====================================================================
    ("oc_001", "open_close", "Sub($close,$open)",
     "Daily close-open (body direction)", "GTJA191"),
    ("oc_002", "open_close", "Div(Sub($close,$open),$open)",
     "Intraday return percentage", "GTJA191"),
    ("oc_003", "open_close", "Div(Sub($open,Ref($close,1d)),Ref($close,1d))",
     "Overnight gap return", "WQ101"),
    ("oc_004", "open_close", "Mean(Sub($open,Ref($close,1d)),20d)",
     "20-day avg overnight gap", "HTSY"),
    ("oc_005", "open_close", "Mean(Sub($close,$open),5d)",
     "5-day avg intraday direction", "GTJA191"),
    ("oc_006", "open_close", "Mean(Abs(Sub($close,$open)),20d)",
     "20-day avg body magnitude", "SWS"),
    ("oc_007", "open_close", "Delta($open,5d)",
     "5-day open price change", "WQ101"),
    ("oc_008", "open_close", "Div($open,$close)",
     "Open-to-close ratio", "GTJA191"),
    ("oc_009", "open_close", "Std(Sub($close,$open),20d)",
     "20-day std of intraday move", "HTSY"),
    ("oc_010", "open_close", "Div(Sub($close,$open),Sub($high,$low))",
     "Body fraction of range", "GTJA191"),

    # ====================================================================
    # CORRELATION / COVARIANCE  (5)
    # ====================================================================
    ("cv_001", "cross_signal", "Corr($volume,$vwap,20d)",
     "20-day volume-VWAP correlation", "WQ101"),
    ("cv_002", "cross_signal", "Cov($close,$volume,20d)",
     "20-day close-volume covariance", "GTJA191"),
    ("cv_003", "cross_signal", "Corr($high,$low,20d)",
     "20-day high-low correlation (range stability)", "SWS"),
    ("cv_004", "cross_signal", "Corr($open,$close,20d)",
     "20-day open-close correlation", "GTJA191"),
    ("cv_005", "cross_signal", "Mul(-1.0,Corr(Delta($close,1d),$volume,20d))",
     "Negative return-volume correlation (mean reversion)", "HTSY"),

    # ====================================================================
    # COMPOUND / GTJA-191 INSPIRED  (5)
    # ====================================================================
    ("cp_001", "compound", "Mul(Corr($volume,$close,10d),Delta($close,5d))",
     "Volume-confirmed momentum (GTJA-style)", "GTJA191"),
    ("cp_002", "compound", "Mul(-1.0,Delta(Div($close,Mean($close,20d)),5d))",
     "Reversal of MA-normalized momentum", "WQ101"),
    ("cp_003", "compound", "Div(Mul(-1.0,Delta($close,1d)),Std($close,5d))",
     "Volatility-adjusted 1-day reversal", "WQ101"),
    ("cp_004", "compound", "Mul(Sub(Div($volume,Mean($volume,5d)),1.0),Delta($close,5d))",
     "Volume-anomaly-weighted momentum", "HTSY"),
    ("cp_005", "compound", "Mul(-1.0,Div(Sub($close,Mean($close,40d)),Mean($close,40d)))",
     "40-day mean reversion (long-term)", "GTJA191"),
]


def main() -> None:
    parser = build_parser()
    out = []
    failures = []
    for fid, family, expr, desc, src in FACTORS:
        try:
            parsed = parser.parse(expr)
        except (ExpressionParsingError, AssertionError, ValueError, KeyError) as e:
            failures.append((fid, expr, str(e)))
            continue
        # Round-trip the parsed expression back to string for canonical form
        out.append({
            "id": fid,
            "family": family,
            "expr": expr,
            "canonical": str(parsed),
            "description": desc,
            "source": src,
        })

    print(f"Total declared: {len(FACTORS)}")
    print(f"Successfully parsed: {len(out)}")
    print(f"Failed: {len(failures)}")
    for fid, expr, err in failures:
        print(f"  FAIL {fid}: {err}")
        print(f"       expr = {expr}")

    if failures:
        sys.exit(1)

    # Family stats
    from collections import Counter
    cnt = Counter(f["family"] for f in out)
    print("\nBy family:")
    for fam, n in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"  {fam:>15s} : {n}")

    cnt_src = Counter(f["source"] for f in out)
    print("\nBy source:")
    for src, n in sorted(cnt_src.items(), key=lambda x: -x[1]):
        print(f"  {src:>10s} : {n}")

    out_path = _ROOT / "data" / "factor_library.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWritten: {out_path}  ({len(out)} factors)")


if __name__ == "__main__":
    main()
