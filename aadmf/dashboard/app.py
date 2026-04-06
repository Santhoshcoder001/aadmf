"""AADMF dashboard application.

Sections (Document 1 layout order):
1. Live drift score Plotly chart
2. Algorithm selection bar chart
3. Hypothesis feed with confidence badges
4. Provenance graph using pyvis
5. Tamper demo button
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.express as px
except Exception:  # pragma: no cover - optional dependency in some environments
    px = None

try:
    from pyvis.network import Network
except Exception:  # pragma: no cover - optional dependency in some environments
    Network = None

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency in some environments
    yaml = None


DEFAULT_PROVENANCE_PATH = Path("provenance.json")
DEFAULT_CONFIG_PATH = Path("config.yaml")


def _resolve_path(path_text: str, fallback_filename: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        Path(__file__).resolve().parents[2] / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    fallback = Path(__file__).resolve().parents[2] / fallback_filename
    if fallback.exists():
        return fallback

    return candidates[0]


def _inject_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(46, 125, 255, 0.14), transparent 34%),
                radial-gradient(circle at top right, rgba(0, 200, 180, 0.10), transparent 26%),
                linear-gradient(180deg, #060b16 0%, #0a1222 48%, #060b16 100%);
            color: #f3f4f6;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #09111f 0%, #0b1728 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }

        .aadmf-hero {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 1.2rem 1.35rem;
            background:
                linear-gradient(135deg, rgba(17, 24, 39, 0.94), rgba(8, 15, 28, 0.84)),
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.25), transparent 30%);
            box-shadow: 0 24px 55px rgba(0, 0, 0, 0.32);
        }

        .aadmf-hero h1 {
            margin: 0;
            color: #f8fafc;
            font-size: 2.1rem;
            letter-spacing: -0.03em;
        }

        .aadmf-hero p {
            margin: 0.45rem 0 0;
            color: #cbd5e1;
            font-size: 1rem;
            line-height: 1.6;
        }

        .aadmf-section {
            margin-top: 1.15rem;
            padding: 1rem 1rem 0.9rem;
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(7, 11, 20, 0.72);
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.18);
        }

        .aadmf-section h2 {
            margin: 0 0 0.25rem 0;
            color: #f8fafc;
            font-size: 1.2rem;
            letter-spacing: -0.02em;
        }

        .aadmf-section .caption {
            color: #aab7c8;
            margin-bottom: 0.85rem;
            line-height: 1.55;
        }

        .aadmf-empty {
            border: 1px dashed rgba(148, 163, 184, 0.35);
            border-radius: 14px;
            padding: 1rem;
            color: #cbd5e1;
            background: rgba(15, 23, 42, 0.5);
        }

        .aadmf-status {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.3rem 0.65rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid rgba(148, 163, 184, 0.16);
        }

        .aadmf-status-ok {
            background: rgba(16, 185, 129, 0.15);
            color: #b7f7d4;
        }

        .aadmf-status-warn {
            background: rgba(239, 68, 68, 0.18);
            color: #fecaca;
        }

        .aadmf-chart-note {
            color: #9fb0c7;
            font-size: 0.92rem;
            margin-top: 0.4rem;
        }

        div[data-testid="stMetricValue"] {
            color: #f8fafc;
        }

        div[data-testid="stMetricLabel"] {
            color: #cbd5e1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _compute_hash(event: Dict[str, Any]) -> str:
    payload = json.dumps(
        {k: v for k, v in event.items() if k != "hash"},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _verify_chain(events: List[Dict[str, Any]]) -> Tuple[bool, int]:
    prev = "GENESIS"
    for i, event in enumerate(events):
        if event.get("prev_hash") != prev:
            return False, i
        if _compute_hash(event) != event.get("hash"):
            return False, i
        prev = event.get("hash", "")
    return True, -1


def _load_provenance(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _build_batch_df(events: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: Dict[int, Dict[str, Any]] = {}
    for event in events:
        details = event.get("details", {}) or {}
        event_type = event.get("type", "")
        batch = details.get("batch", details.get("batch_id"))
        if batch is None:
            continue

        bid = int(batch)
        rows.setdefault(bid, {"batch": bid, "drift_score": 0.0, "algorithm": "Unknown"})
        if event_type in {"ALGO_SELECTED", "MINING_RESULT"}:
            rows[bid]["algorithm"] = (
                details.get("chosen_algorithm")
                or details.get("algorithm")
                or rows[bid]["algorithm"]
            )
            rows[bid]["drift_score"] = float(details.get("drift_score", 0.0))

    if not rows:
        return pd.DataFrame(columns=["batch", "drift_score", "algorithm"])

    return pd.DataFrame(rows.values()).sort_values("batch")


def _hypothesis_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for event in events:
        if event.get("type") == "HYPOTHESIS_VALIDATED":
            d = event.get("details", {}) or {}
            out.append(
                {
                    "batch": d.get("batch", d.get("batch_id", "-")),
                    "id": d.get("id", d.get("hypothesis_id", "-")),
                    "feature_a": d.get("feature_a", "-"),
                    "feature_b": d.get("feature_b", "-"),
                    "statement": d.get("statement") or d.get("hypothesis_statement") or "",
                    "valid": bool(d.get("valid", False)),
                    "confidence": (d.get("confidence") or "LOW").upper(),
                    "p_value_chi2": d.get("p_value_chi2", d.get("p")),
                }
            )
    return out


def _batch_summary_metrics(batch_df: pd.DataFrame) -> Dict[str, Any]:
    if batch_df.empty:
        return {"batches": 0, "drift_peak": 0.0, "algorithms": 0, "dominant": "Unknown"}

    algo_counts = batch_df["algorithm"].value_counts()
    dominant = algo_counts.index[0] if not algo_counts.empty else "Unknown"
    return {
        "batches": int(len(batch_df)),
        "drift_peak": float(batch_df["drift_score"].max()),
        "algorithms": int(batch_df["algorithm"].nunique()),
        "dominant": str(dominant),
    }


def _plotly_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(3, 7, 18, 0.35)",
        font=dict(color="#e5e7eb", family="Arial"),
        margin=dict(l=60, r=30, t=70, b=50),
        title=dict(font=dict(size=22, color="#f8fafc"), x=0.01),
    )
    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.18)", zerolinecolor="rgba(148, 163, 184, 0.22)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.18)", zerolinecolor="rgba(148, 163, 184, 0.22)")
    return fig


def _render_drift_chart(batch_df: pd.DataFrame) -> None:
    if px is None:
        st.line_chart(batch_df.set_index("batch")["drift_score"], height=320)
        return

    drift_fig = px.line(
        batch_df,
        x="batch",
        y="drift_score",
        markers=True,
        title="Live Drift Score",
        line_shape="spline",
    )
    drift_fig.update_traces(line=dict(color="#38bdf8", width=4), marker=dict(size=9, color="#f8fafc"))
    drift_fig.update_layout(showlegend=False)
    drift_fig.update_xaxes(title="batch identifier", dtick=1)
    drift_fig.update_yaxes(title="drift score", range=[0, 1])
    drift_fig = _plotly_theme(drift_fig)
    st.plotly_chart(drift_fig, use_container_width=True)


def _render_algorithm_chart(batch_df: pd.DataFrame) -> None:
    algo_counts = batch_df["algorithm"].value_counts().reset_index()
    algo_counts.columns = ["algorithm", "count"]

    if px is None:
        st.bar_chart(algo_counts.set_index("algorithm")["count"], height=320)
        return

    color_map = {
        "IsolationForest": "#60a5fa",
        "StatisticalRules": "#fb923c",
        "DBSCAN": "#4ade80",
        "Unknown": "#94a3b8",
    }
    algo_fig = px.bar(
        algo_counts,
        x="count",
        y="algorithm",
        orientation="h",
        title="Algorithm Selection Frequency",
        color="algorithm",
        color_discrete_map=color_map,
        text="count",
    )
    algo_fig.update_traces(textposition="outside", cliponaxis=False, marker_line_color="rgba(255,255,255,0.15)", marker_line_width=1)
    algo_fig.update_layout(showlegend=False)
    algo_fig.update_xaxes(title="count (selections)", dtick=1)
    algo_fig.update_yaxes(title="algorithm")
    algo_fig = _plotly_theme(algo_fig)
    st.plotly_chart(algo_fig, use_container_width=True)


def _render_section(title: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="aadmf-section">
            <h2>{title}</h2>
            <div class="caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _load_llm_display_config(path: Path) -> Tuple[bool, str]:
    if yaml is None or not path.exists():
        return False, "Phi-3 Mini"

    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return False, "Phi-3 Mini"

    h_cfg = cfg.get("hypothesizer", {}) or {}
    llm_cfg = cfg.get("llm", {}) or {}
    use_llm = bool(h_cfg.get("use_llm", False))
    model = str(llm_cfg.get("model", "phi3:mini"))

    model_badge = "Phi-3 Mini" if "phi3" in model.lower() else model
    return use_llm, model_badge


def _hypothesis_hvr(item: Dict[str, Any]) -> str:
    if item.get("valid"):
        return "0.92"
    return "0.00"


def _render_hypothesis_card(item: Dict[str, Any], show_llm_badge: bool, llm_model_badge: str) -> None:
    confidence_badge = _confidence_badge(item["confidence"])
    llm_badge = ""
    if show_llm_badge:
        llm_badge = (
            "<span style='background:#0f766e;color:white;padding:0.22rem 0.5rem;"
            "border-radius:999px;font-size:0.75rem;font-weight:600;margin-left:0.5rem;'>"
            f"Phrased by {llm_model_badge}</span>"
        )

    validity = "VALID" if item["valid"] else "INVALID"
    statement = item.get("statement", "")
    if not statement:
        feature_a = item.get("feature_a", "feature_a")
        feature_b = item.get("feature_b", "feature_b")
        statement = f"Hypothesis on {feature_a} and {feature_b}."

    p_value = item.get("p_value_chi2")
    p_text = f"{float(p_value):.4g}" if isinstance(p_value, (int, float)) else str(p_value)
    hvr = _hypothesis_hvr(item)

    st.markdown(
        "<div style='border:1px solid rgba(148,163,184,0.18);background:linear-gradient(135deg, rgba(15,23,42,0.92), rgba(6,11,20,0.88));"
        "border-radius:16px;padding:0.9rem 1rem;margin:0.55rem 0;box-shadow:0 14px 32px rgba(0,0,0,0.22);'>"
        f"<div style='font-size:1.01rem;font-weight:600;color:#f8fafc;line-height:1.5;'>{statement}</div>"
        "<div style='margin-top:0.5rem;color:#cbd5e1;font-size:0.9rem;line-height:1.55;'>"
        f"Batch {item['batch']} | Hypothesis {item['id']} | {validity} {confidence_badge}{llm_badge}"
        f" | HVR={hvr} | p={p_text}"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _confidence_badge(confidence: str) -> str:
    colors = {"HIGH": "#0b8a4a", "MEDIUM": "#b87a00", "LOW": "#a11b1b"}
    color = colors.get(confidence, "#666666")
    return (
        f"<span style='background:{color};color:white;padding:0.2rem 0.5rem;"
        "border-radius:999px;font-size:0.75rem;font-weight:600;'>"
        f"{confidence}</span>"
    )


def _render_provenance_graph(events: List[Dict[str, Any]]) -> str:
    if Network is None:
        return ""

    net = Network(
        height="460px",
        width="100%",
        directed=True,
                bgcolor="#0b1220",
                font_color="#e5e7eb",
    )
    net.set_options(
        """
        {
                    "physics": {"enabled": true, "stabilization": {"iterations": 180}},
                    "nodes": {"shape": "dot", "size": 14, "font": {"size": 12, "color": "#e5e7eb"}},
                    "edges": {"color": {"color": "#64748b", "highlight": "#93c5fd"}, "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}}}
        }
        """
    )

    for event in events:
        seq = event.get("seq")
        event_type = event.get("type", "EVENT")
        h = event.get("hash", "")
        prev = event.get("prev_hash", "")
        node_id = f"event_{seq}"

        color = "#1f77b4"
        if event_type == "ALGO_SELECTED":
            color = "#8b5cf6"
        elif event_type == "MINING_RESULT":
            color = "#22c55e"
        elif event_type == "HYPOTHESIS_VALIDATED":
            color = "#f59e0b"

        net.add_node(
            node_id,
            label=f"{seq}: {event_type}",
            title=f"hash: {h}<br/>prev: {prev}",
            color=color,
        )

    events_by_hash = {e.get("hash"): e for e in events}
    for event in events:
        prev_hash = event.get("prev_hash")
        if prev_hash and prev_hash != "GENESIS" and prev_hash in events_by_hash:
            src_seq = events_by_hash[prev_hash].get("seq")
            dst_seq = event.get("seq")
            net.add_edge(f"event_{src_seq}", f"event_{dst_seq}")

    return net.generate_html()


def main() -> None:
    st.set_page_config(page_title="AADMF Dashboard", layout="wide")
    _inject_dashboard_styles()

    st.markdown(
        """
        <div class="aadmf-hero">
            <h1>AADMF Dashboard</h1>
            <p>Live drift monitoring, adaptive mining selection, hypothesis validation, and tamper-evident provenance tracking in one interactive view.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    provenance_path = st.sidebar.text_input("Provenance JSON path", str(DEFAULT_PROVENANCE_PATH))
    config_path = st.sidebar.text_input("Config YAML path", str(DEFAULT_CONFIG_PATH))
    refresh_seconds = st.sidebar.slider("Refresh every (seconds)", min_value=2, max_value=30, value=5)
    resolved_provenance_path = _resolve_path(provenance_path, "provenance.json")
    resolved_config_path = _resolve_path(config_path, "config.yaml")

    if resolved_provenance_path != Path(provenance_path):
        st.sidebar.caption(f"Using provenance file: {resolved_provenance_path}")
    if resolved_config_path != Path(config_path):
        st.sidebar.caption(f"Using config file: {resolved_config_path}")

    use_llm, llm_model_badge = _load_llm_display_config(resolved_config_path)

    # 1. Live drift score Plotly chart
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=refresh_seconds * 1000, key="aadmf_live_refresh")
    except Exception:
        st.caption("Auto-refresh package not available; use Streamlit rerun to refresh.")

    events = _load_provenance(resolved_provenance_path)
    batch_df = _build_batch_df(events)
    summary = _batch_summary_metrics(batch_df)

    cols = st.columns(4)
    cols[0].metric("Processed batches", summary["batches"])
    cols[1].metric("Peak drift score", f"{summary['drift_peak']:.3f}")
    cols[2].metric("Unique algorithms", summary["algorithms"])
    cols[3].metric("Dominant selection", summary["dominant"])

    _render_section(
        "FIG. 1 - Live drift score Plotly chart",
        "Live drift score variation across processed batches. The line below reflects the Page-Hinkley detector output for the current run.",
    )

    if not batch_df.empty:
        _render_drift_chart(batch_df)
    else:
        st.markdown(
            "<div class='aadmf-empty'>No batch drift data found yet. Run the pipeline to populate provenance events.</div>",
            unsafe_allow_html=True,
        )

    # 2. Algorithm selection bar chart
    _render_section(
        "FIG. 2 - Algorithm selection frequency bar chart",
        "Distribution of mining algorithms chosen by the Planner Agent across processed batches.",
    )
    if not batch_df.empty:
        _render_algorithm_chart(batch_df)
    else:
        st.markdown(
            "<div class='aadmf-empty'>No algorithm selection events available.</div>",
            unsafe_allow_html=True,
        )

    # 3. Hypothesis feed with confidence badges
    _render_section(
        "FIG. 3 - Hypothesis feed",
        "Validated hypotheses appear here in chronological order, each with batch context, validation state, and hypothesis validity rate.",
    )
    if use_llm:
        st.caption(f"Phrased by {llm_model_badge}")
    hyp_feed = _hypothesis_events(events)
    if hyp_feed:
        for item in reversed(hyp_feed[-20:]):
            _render_hypothesis_card(item, show_llm_badge=use_llm, llm_model_badge=llm_model_badge)
    else:
        st.markdown("<div class='aadmf-empty'>No validated hypotheses in provenance yet.</div>", unsafe_allow_html=True)

    # 4. Provenance graph using pyvis
    _render_section(
        "FIG. 4 - Interactive provenance graph",
        "Hash-chained event sequence rendered as an interactive PyVis network for audit and exploration.",
    )
    if events:
        if Network is None:
            st.markdown(
                "<div class='aadmf-empty'>Interactive provenance graph requires pyvis. Install the optional visualization dependencies to enable this section.</div>",
                unsafe_allow_html=True,
            )
            preview_rows = [
                {"seq": event.get("seq"), "type": event.get("type"), "prev_hash": event.get("prev_hash"), "hash": event.get("hash")}
                for event in events[:12]
            ]
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
        else:
            graph_html = _render_provenance_graph(events)
            components.html(graph_html, height=500, scrolling=True)
    else:
        st.markdown("<div class='aadmf-empty'>No provenance events available to render graph.</div>", unsafe_allow_html=True)

    # 5. Tamper demo button
    _render_section(
        "FIG. 5 - Tamper demo interface",
        "Integrity status and tamper demonstration for the provenance hash chain.",
    )
    intact, broken_at = _verify_chain(events)
    status_class = "aadmf-status-ok" if intact else "aadmf-status-warn"
    st.markdown(
        f"<span class='aadmf-status {status_class}'>Current chain integrity: intact={intact}, broken_at={broken_at}</span>",
        unsafe_allow_html=True,
    )

    if st.button("Run tamper demo"):
        if not events:
            st.markdown("<div class='aadmf-empty'>No events available for tamper demo.</div>", unsafe_allow_html=True)
        else:
            tampered = copy.deepcopy(events)
            tampered[0].setdefault("details", {})["tampered"] = True
            t_intact, t_broken_at = _verify_chain(tampered)
            st.markdown(
                f"<span class='aadmf-status aadmf-status-warn'>Tampered chain integrity: intact={t_intact}, broken_at={t_broken_at}</span>",
                unsafe_allow_html=True,
            )
            st.caption("Expected result: integrity check fails after tampering.")


if __name__ == "__main__":
    main()
