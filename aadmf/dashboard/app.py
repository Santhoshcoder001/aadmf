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
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency in some environments
    yaml = None


DEFAULT_PROVENANCE_PATH = Path("provenance.json")
DEFAULT_CONFIG_PATH = Path("config.yaml")


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
        batch = details.get("batch")
        if batch is None:
            continue

        bid = int(batch)
        rows.setdefault(bid, {"batch": bid, "drift_score": 0.0, "algorithm": "Unknown"})
        if event_type == "ALGO_SELECTED":
            rows[bid]["algorithm"] = details.get("algorithm", "Unknown")
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

    st.markdown(
        "<div style='border:1px solid #e5e7eb;border-left:5px solid #0ea5e9;"
        "border-radius:12px;padding:0.75rem 0.9rem;margin:0.45rem 0;'>"
        f"<div style='font-size:1.02rem;font-weight:600;color:#111827;line-height:1.45;'>{statement}</div>"
        "<div style='margin-top:0.45rem;color:#374151;font-size:0.88rem;'>"
        f"Batch {item['batch']} | Hypothesis {item['id']} | {validity} {confidence_badge}{llm_badge}"
        f" | p={p_text}"
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
    net = Network(
        height="460px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#1f2937",
    )
    net.set_options(
        """
        {
          "physics": {"enabled": true, "stabilization": {"iterations": 150}},
          "nodes": {"shape": "dot", "size": 12, "font": {"size": 12}},
          "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 0.6}}}
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
            color = "#9467bd"
        elif event_type == "MINING_RESULT":
            color = "#2ca02c"
        elif event_type == "HYPOTHESIS_VALIDATED":
            color = "#ff7f0e"

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
    st.title("AADMF Dashboard")

    provenance_path = st.sidebar.text_input("Provenance JSON path", str(DEFAULT_PROVENANCE_PATH))
    config_path = st.sidebar.text_input("Config YAML path", str(DEFAULT_CONFIG_PATH))
    refresh_seconds = st.sidebar.slider("Refresh every (seconds)", min_value=2, max_value=30, value=5)
    use_llm, llm_model_badge = _load_llm_display_config(Path(config_path))

    # 1. Live drift score Plotly chart
    st.header("1. Live drift score Plotly chart")
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=refresh_seconds * 1000, key="aadmf_live_refresh")
    except Exception:
        st.caption("Auto-refresh package not available; use Streamlit rerun to refresh.")

    events = _load_provenance(Path(provenance_path))
    batch_df = _build_batch_df(events)

    if not batch_df.empty:
        drift_fig = px.line(
            batch_df,
            x="batch",
            y="drift_score",
            markers=True,
            title="Drift Score Over Batches",
        )
        drift_fig.update_yaxes(range=[0, 1])
        st.plotly_chart(drift_fig, use_container_width=True)
    else:
        st.info("No batch drift data found yet. Run the pipeline to populate provenance events.")

    # 2. Algorithm selection bar chart
    st.header("2. Algorithm selection bar chart")
    if not batch_df.empty:
        algo_counts = batch_df["algorithm"].value_counts().reset_index()
        algo_counts.columns = ["algorithm", "count"]
        algo_fig = px.bar(algo_counts, x="algorithm", y="count", title="Algorithm Selection Frequency")
        st.plotly_chart(algo_fig, use_container_width=True)
    else:
        st.info("No algorithm selection events available.")

    # 3. Hypothesis feed with confidence badges
    st.header("3. Hypothesis feed")
    if use_llm:
        st.caption(f"Phrased by {llm_model_badge}")
    hyp_feed = _hypothesis_events(events)
    if hyp_feed:
        for item in reversed(hyp_feed[-20:]):
            _render_hypothesis_card(item, show_llm_badge=use_llm, llm_model_badge=llm_model_badge)
    else:
        st.info("No validated hypotheses in provenance yet.")

    # 4. Provenance graph using pyvis
    st.header("4. Provenance graph using pyvis")
    if events:
        graph_html = _render_provenance_graph(events)
        components.html(graph_html, height=500, scrolling=True)
    else:
        st.info("No provenance events available to render graph.")

    # 5. Tamper demo button
    st.header("5. Tamper demo button")
    intact, broken_at = _verify_chain(events)
    st.write(f"Current chain integrity: intact={intact}, broken_at={broken_at}")

    if st.button("Run tamper demo"):
        if not events:
            st.warning("No events available for tamper demo.")
        else:
            tampered = copy.deepcopy(events)
            tampered[0].setdefault("details", {})["tampered"] = True
            t_intact, t_broken_at = _verify_chain(tampered)
            st.error(f"Tampered chain integrity: intact={t_intact}, broken_at={t_broken_at}")
            st.caption("Expected result: integrity check fails after tampering.")


if __name__ == "__main__":
    main()
