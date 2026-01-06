import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

DATA_DIR = Path("data")

LIVE_STATE_PATH = DATA_DIR / "live_state.json"
SUMMARY_HISTORY_PATH = DATA_DIR / "summary_history.jsonl"
DECISIONS_LOG_PATH = DATA_DIR / "decisions.jsonl"
ALERTS_LOG_PATH = DATA_DIR / "alert_events.jsonl"  # or _stress.jsonl for stress mode


# -------------------------
# Helpers
# -------------------------
def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def to_dt(series: pd.Series) -> pd.Series:
    # ts is unix seconds
    return pd.to_datetime(series, unit="s", errors="coerce")


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            continue

        s = df[c]


        if isinstance(s, pd.DataFrame):
            # take the first occurrence
            s = s.iloc[:, 0]

        df[c] = pd.to_numeric(s, errors="coerce")

    return df



def build_discrete_risk_colorscale():
    # 0=OK, 1=Warning, 2=Critical
    # Plotly discrete colors via continuous mapping segments
    return [
        [0.0, "#2ecc71"], [0.333, "#2ecc71"],     # green
        [0.333, "#f1c40f"], [0.666, "#f1c40f"],   # yellow
        [0.666, "#e74c3c"], [1.0, "#e74c3c"],     # red
    ]


def normalize_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            x = pd.to_numeric(out[c], errors="coerce")
            mu = x.mean()
            sd = x.std()
            if sd is None or sd == 0 or math.isnan(sd):
                out[c] = np.nan
            else:
                out[c] = (x - mu) / sd
    return out


# -------------------------
# Page
# -------------------------
st.set_page_config(page_title="IoT Real-Time Dashboard", layout="wide")
st.title("IoT Real-Time Dashboard")

# Auto refresh every 1s (change if you want)
st_autorefresh(interval=1000, key="refresh")

# Load artifacts
live = safe_read_json(LIVE_STATE_PATH) or {}
summary_df = safe_read_jsonl(SUMMARY_HISTORY_PATH)
decisions_df = safe_read_jsonl(DECISIONS_LOG_PATH)
alerts_df = safe_read_jsonl(ALERTS_LOG_PATH)

# Preprocess
if not summary_df.empty and "ts" in summary_df.columns:
    summary_df["dt"] = to_dt(summary_df["ts"])
    # flatten some nested fields if present
    # expected: summary["edge"]["in"], summary["edge"]["arrays_out"], etc.
    def unpack_summary_row(row):
        edge = row.get("edge", {}) if isinstance(row.get("edge"), dict) else {}
        lat = row.get("latency", {}) if isinstance(row.get("latency"), dict) else {}
        infer = lat.get("infer", {}) if isinstance(lat.get("infer"), dict) else {}
        e2e = lat.get("e2e", {}) if isinstance(lat.get("e2e"), dict) else {}
        return pd.Series({
            "edge_in": edge.get("in"),
            "arrays_out": edge.get("arrays_out"),
            "dropped": edge.get("dropped"),
            "late": edge.get("late"),
            "corrected_fields": edge.get("corrected_fields"),
            "compression": edge.get("compression"),
            "alerts_emitted": row.get("alerts_emitted"),
            "infer_avg": infer.get("avg_ms"),
            "infer_p95": infer.get("p95_ms"),
            "e2e_avg": e2e.get("avg_ms"),
            "e2e_p95": e2e.get("p95_ms"),
        })
    unpacked = summary_df.apply(unpack_summary_row, axis=1)
    summary_df = pd.concat([summary_df, unpacked], axis=1)
    summary_df = summary_df.loc[:, ~summary_df.columns.duplicated()]

    summary_df = coerce_numeric(
        summary_df,
        ["edge_in", "arrays_out", "dropped", "late", "corrected_fields", "compression",
         "alerts_emitted", "infer_avg", "infer_p95", "e2e_avg", "e2e_p95"]
    )

if not decisions_df.empty:
    # expected columns: ts, patient_id, risk_level, latency_ms, e2e_ms, + optional vitals stats
    if "ts" in decisions_df.columns:
        decisions_df["dt"] = to_dt(decisions_df["ts"])
    decisions_df = coerce_numeric(decisions_df, ["patient_id", "latency_ms", "e2e_ms"])
    # risk_level may be string/int
    if "risk_level" in decisions_df.columns:
        decisions_df["risk_level_num"] = pd.to_numeric(decisions_df["risk_level"], errors="coerce")

if not alerts_df.empty:
    if "ts" in alerts_df.columns:
        alerts_df["dt"] = to_dt(alerts_df["ts"])
    alerts_df = coerce_numeric(alerts_df, ["patient_id"])
    if "risk_level" in alerts_df.columns:
        alerts_df["risk_level_num"] = pd.to_numeric(alerts_df["risk_level"], errors="coerce")


# -------------------------
# Status / KPIs
# -------------------------
with st.container():
    c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.2])

    live_ok = "OK" if live else "NO DATA"
    live_time = None
    if live and "ts" in live:
        live_time = pd.to_datetime(live["ts"], unit="s", errors="coerce")

    summary_lines = 0 if summary_df.empty else len(summary_df)
    decision_lines = 0 if decisions_df.empty else len(decisions_df)
    alert_lines = 0 if alerts_df.empty else len(alerts_df)

    c1.metric("Status", live_ok)
    c2.metric("live_state time", str(live_time.time()) if live_time is not None else "-")
    c3.metric("summary lines", summary_lines)
    c4.metric("decisions lines", decision_lines)
    c5.metric("alerts lines", alert_lines)

    if live and "line" in live:
        st.code(live["line"], language="text")

# Fault KPI block (from live state if present)
if live and isinstance(live.get("edge"), dict):
    edge = live["edge"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Raw events (in)", edge.get("in", 0))
    k2.metric("Arrays out", edge.get("arrays_out", 0))
    k3.metric("Late events", edge.get("late", 0))
    k4.metric("Dropped", edge.get("dropped", 0))
    k5.metric("Corrected fields", edge.get("corrected_fields", 0))

st.divider()

# -------------------------
# Sankey: Raw -> Edge -> Cloud -> Alerts
# -------------------------
st.subheader("Flow Visualization (Sankey): Raw → Edge → Cloud → Alerts")

raw = int(live.get("edge", {}).get("in", 0)) if live else 0
arrays = int(live.get("edge", {}).get("arrays_out", 0)) if live else 0
decisions_cnt = len(decisions_df) if not decisions_df.empty else 0
alerts_cnt = int(live.get("alerts_emitted", 0)) if live else (len(alerts_df) if not alerts_df.empty else 0)

labels = ["Raw events", "Edge arrays", "Cloud decisions", "Alerts"]
source = [0, 1, 2]
target = [1, 2, 3]
values = [max(raw, 1), max(arrays, 1), max(alerts_cnt, 1)]

fig_sankey = go.Figure(
    data=[
        go.Sankey(
            node=dict(label=labels, pad=15, thickness=20),
            link=dict(source=source, target=target, value=values),
        )
    ]
)
fig_sankey.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_sankey, use_container_width=True)

st.caption("Sankey shows reduction of raw telemetry on the edge (window aggregation), then cloud decisions and alerts.")

st.divider()

# -------------------------
# Risk heatmap (patient x time)
# -------------------------
st.subheader("Risk Heatmap (patient × time)")

if decisions_df.empty or "patient_id" not in decisions_df.columns or "risk_level_num" not in decisions_df.columns:
    st.info("No decisions data yet (decisions.jsonl). Run the multi_demo and ensure decisions are logged.")
else:
    # bucket time into 2-second bins for heatmap stability
    df = decisions_df.dropna(subset=["dt", "patient_id", "risk_level_num"]).copy()
    df["bucket"] = df["dt"].dt.floor("2S")

    # Pivot to matrix patient x time
    mat = df.pivot_table(
        index="patient_id",
        columns="bucket",
        values="risk_level_num",
        aggfunc="last",
    ).sort_index()

    heat = go.Figure(
        data=go.Heatmap(
            z=mat.values,
            x=mat.columns.astype(str),
            y=mat.index.astype(int),
            colorscale=build_discrete_risk_colorscale(),
            zmin=0,
            zmax=2,
            colorbar=dict(
                title="Risk",
                tickmode="array",
                tickvals=[0, 1, 2],
                ticktext=["0 OK", "1 Warn", "2 Crit"],
            ),
        )
    )
    heat.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(heat, use_container_width=True)

st.divider()

# -------------------------
# Patient monitor: vitals + risk + alert markers
# -------------------------
st.subheader("Medical Monitor (selected patient): Vitals + Risk + Alerts")

if decisions_df.empty or "patient_id" not in decisions_df.columns:
    st.info("No decisions data yet.")
else:
    patients = sorted([int(x) for x in decisions_df["patient_id"].dropna().unique().tolist() if x >= 0])
    if not patients:
        st.info("No patient IDs in decisions log.")
    else:
        left, right = st.columns([1.2, 3.0])
        with left:
            pid = st.selectbox("Patient", patients, index=0)
            mode = st.radio("Vitals scale", ["Dual-axis", "Normalized"], index=0)
            window_sec = st.slider("Show last N seconds", 30, 600, 120, step=10)

        dfp = decisions_df[(decisions_df["patient_id"] == pid)].dropna(subset=["dt"]).copy()
        if dfp.empty:
            st.warning("No data for selected patient yet.")
        else:
            tmax = dfp["dt"].max()
            tmin = tmax - pd.Timedelta(seconds=int(window_sec))
            dfp = dfp[dfp["dt"] >= tmin].copy()

            # Try to find vitals columns (prefer means from info arrays)
            candidate_cols = [
                ("hr_mean", "HR mean"),
                ("spo2_mean", "SpO2 mean"),
                ("sbp_mean", "SBP mean"),
                ("dbp_mean", "DBP mean"),
                # fallback if you log raw
                ("heart_rate", "HR"),
                ("spo2", "SpO2"),
                ("sbp", "SBP"),
                ("dbp", "DBP"),
            ]
            present = [(c, label) for (c, label) in candidate_cols if c in dfp.columns]
            vital_cols = [c for c, _ in present]

            # Alerts for this patient
            ap = pd.DataFrame()
            if not alerts_df.empty and "patient_id" in alerts_df.columns:
                ap = alerts_df[alerts_df["patient_id"] == pid].dropna(subset=["dt"]).copy()
                ap = ap[(ap["dt"] >= tmin) & (ap["dt"] <= tmax)]

            # Vitals plot
            if vital_cols:
                if mode == "Normalized":
                    plot_df = normalize_cols(dfp, vital_cols)
                    y_title = "Normalized units (z-score)"
                    # single axis
                    fig_v = go.Figure()
                    for c, label in present:
                        if c in plot_df.columns:
                            fig_v.add_trace(go.Scatter(x=plot_df["dt"], y=plot_df[c], mode="lines", name=label))
                else:
                    # Dual-axis: HR+SpO2 on left, BP on right (if present)
                    fig_v = make_subplots(specs=[[{"secondary_y": True}]])
                    left_set = {"hr_mean", "heart_rate", "spo2_mean", "spo2"}
                    right_set = {"sbp_mean", "dbp_mean", "sbp", "dbp"}
                    for c, label in present:
                        if c in dfp.columns:
                            sec = c in right_set
                            fig_v.add_trace(
                                go.Scatter(x=dfp["dt"], y=dfp[c], mode="lines", name=label),
                                secondary_y=sec,
                            )
                    fig_v.update_yaxes(title_text="HR / SpO2", secondary_y=False)
                    fig_v.update_yaxes(title_text="Blood pressure", secondary_y=True)
                    y_title = ""

                # Alert markers on vitals
                if not ap.empty:
                    for _, r in ap.iterrows():
                        fig_v.add_vline(
                            x=r["dt"],
                            line_width=2,
                            line_dash="dash",
                            line_color="red",
                            opacity=0.7,
                        )

                fig_v.update_layout(
                    height=320,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="time",
                    yaxis_title=y_title,
                    legend=dict(orientation="h"),
                )
                st.plotly_chart(fig_v, use_container_width=True)
            else:
                st.warning("No vitals columns found in decisions.jsonl. Add hr_mean/spo2_mean/sbp_mean/dbp_mean to decisions log (patch below).")

            # Risk plot + alert markers
            if "risk_level_num" in dfp.columns:
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(x=dfp["dt"], y=dfp["risk_level_num"], mode="lines+markers", name="Risk level"))
                fig_r.update_yaxes(
                    tickmode="array",
                    tickvals=[0, 1, 2],
                    ticktext=["0 OK", "1 Warn", "2 Crit"],
                    range=[-0.2, 2.2],
                )

                if not ap.empty:
                    for _, r in ap.iterrows():
                        fig_r.add_vline(
                            x=r["dt"],
                            line_width=2,
                            line_dash="dash",
                            line_color="red",
                            opacity=0.8,
                        )

                fig_r.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="time")
                st.plotly_chart(fig_r, use_container_width=True)

st.divider()

# -------------------------
# Latency distributions + latency over time
# -------------------------
st.subheader("Latency: distributions + trend")

if decisions_df.empty or "latency_ms" not in decisions_df.columns or "e2e_ms" not in decisions_df.columns:
    st.info("No latency data in decisions log yet.")
else:
    dfx = decisions_df.dropna(subset=["latency_ms", "e2e_ms"]).copy()
    c1, c2 = st.columns(2)

    with c1:
        fig_h1 = px.histogram(dfx, x="latency_ms", nbins=25, title="Inference latency (ms)")
        fig_h1.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_h1, use_container_width=True)

    with c2:
        fig_h2 = px.histogram(dfx, x="e2e_ms", nbins=25, title="End-to-end latency (ms)")
        fig_h2.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_h2, use_container_width=True)

    # Trend (rolling)
    if "dt" in dfx.columns:
        dfx = dfx.sort_values("dt")
        dfx["lat_p95_rolling"] = dfx["latency_ms"].rolling(30, min_periods=5).quantile(0.95)
        dfx["e2e_p95_rolling"] = dfx["e2e_ms"].rolling(30, min_periods=5).quantile(0.95)

        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=dfx["dt"], y=dfx["latency_ms"], mode="lines", name="infer (ms)", opacity=0.35))
        fig_t.add_trace(go.Scatter(x=dfx["dt"], y=dfx["e2e_ms"], mode="lines", name="e2e (ms)", opacity=0.35))
        fig_t.add_trace(go.Scatter(x=dfx["dt"], y=dfx["lat_p95_rolling"], mode="lines", name="infer p95 rolling", line=dict(width=3)))
        fig_t.add_trace(go.Scatter(x=dfx["dt"], y=dfx["e2e_p95_rolling"], mode="lines", name="e2e p95 rolling", line=dict(width=3)))
        fig_t.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="time")
        st.plotly_chart(fig_t, use_container_width=True)

st.divider()

# -------------------------
# NEW GRAPH: Throughput + compression trend
# -------------------------
st.subheader("Throughput + Compression (from summary history)")

if summary_df.empty or "dt" not in summary_df.columns:
    st.info("No summary history yet.")
else:
    sdf = summary_df.dropna(subset=["dt", "edge_in", "arrays_out"]).sort_values("dt").copy()
    if len(sdf) >= 2:
        sdf["edge_in_prev"] = sdf["edge_in"].shift(1)
        sdf["arrays_prev"] = sdf["arrays_out"].shift(1)
        sdf["dt_prev"] = sdf["dt"].shift(1)

        dt_sec = (sdf["dt"] - sdf["dt_prev"]).dt.total_seconds()
        sdf["events_per_sec"] = (sdf["edge_in"] - sdf["edge_in_prev"]) / dt_sec
        sdf["arrays_per_sec"] = (sdf["arrays_out"] - sdf["arrays_prev"]) / dt_sec

        fig_thr = go.Figure()
        fig_thr.add_trace(go.Scatter(x=sdf["dt"], y=sdf["events_per_sec"], mode="lines+markers", name="events/sec"))
        fig_thr.add_trace(go.Scatter(x=sdf["dt"], y=sdf["arrays_per_sec"], mode="lines+markers", name="arrays/sec"))
        fig_thr.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="time", yaxis_title="rate")
        st.plotly_chart(fig_thr, use_container_width=True)

    if "compression" in summary_df.columns:
        fig_comp = px.line(summary_df.dropna(subset=["dt", "compression"]), x="dt", y="compression", title="Compression ratio over time")
        fig_comp.update_layout(height=240, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_comp, use_container_width=True)

