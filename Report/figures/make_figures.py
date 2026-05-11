"""Generate the Report/ figures as professional PDF files.

Outputs (all written to Report/figures/):
  fig1_pipeline.pdf   - pipeline / architecture diagram
  fig2_ablation.pdf   - 3-panel macro ROC/PR/F1 across the 5 ablation rows
  fig3_heatmap.pdf    - per-(model x compartment) ROC-AUC heatmap
  fig4_sweep.pdf      - 8-config random search val ROC-AUC bar chart
  fig5_motifs.pdf     - GNNExplainer motif-overlap bars
  fig6_curves.pdf     - ROC + PR overlay curves for two illustrative compartments

All figures use Plotly with kaleido for vector PDF export. The visual
language is consistent across plots: a muted blue palette for non-headline
rows and a warm red highlight (#E45756) for the headline GAT row.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, roc_curve

# -------- paths --------
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS_CLASSICAL = REPO / "runs" / "spec_classical"
RUNS_GNN = REPO / "runs" / "gnn"
OUT = HERE

# -------- shared style --------
PAL = {
    "classic": "#7F7F7F",
    "esmmlp":  "#9ECAE1",
    "gatseq":  "#6BAED6",
    "gatcon":  "#3182BD",
    "gatboth": "#E45756",   # headline highlight
    "neutral": "#4C78A8",
    "miss":    "#C7C7C7",
    "hit":     "#2CA02C",
}
FONT = dict(family="Times New Roman, serif", size=13, color="#222222")
def base_layout(**overrides):
    layout = dict(
        template="simple_white",
        font=FONT,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=60, r=20, t=46, b=60),
    )
    if "margin" in overrides:
        layout["margin"] = overrides.pop("margin")
    layout.update(overrides)
    return layout


def save(fig: go.Figure, name: str, w: int = 760, h: int = 420):
    out = OUT / name
    fig.write_image(str(out), format="pdf", width=w, height=h)
    print(f"  wrote {out.relative_to(REPO)} ({w}x{h})")


# =========================================================================
# FIG 1: pipeline diagram
# =========================================================================
def fig1_pipeline():
    # Hand-laid grid: x in [0, 14], y in [0, 6]. Rectangles + arrows + text.
    fig = go.Figure()

    # block helper
    def block(x0, y0, x1, y1, fill, line, text, text_color="black", italic=False):
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color=line, width=1.2), fillcolor=fill, layer="below")
        fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=text, showarrow=False,
                           font=dict(family=FONT["family"], size=12, color=text_color))

    def arrow(x0, y0, x1, y1, color="#666"):
        fig.add_annotation(ax=x0, ay=y0, x=x1, y=y1,
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1.0,
                           arrowwidth=1.2, arrowcolor=color)

    def group(x0, y0, x1, y1, fill, label):
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color="rgba(0,0,0,0.10)", width=1, dash="dot"),
                      fillcolor=fill, layer="below")
        fig.add_annotation(x=(x0+x1)/2, y=y1-0.18, text=label, showarrow=False,
                           font=dict(family=FONT["family"], size=10, color="#555"),
                           xanchor="center", yanchor="top")

    # background groupings
    group(2.4, 1.5, 8.6, 5.5, "rgba(220,230,250,0.20)", "<i>precomputed once</i>")
    group(8.9, 2.0, 12.6, 4.5, "rgba(255,235,205,0.30)", "<i>trained (~0.3M params)</i>")

    # nodes
    block(0.2, 2.7, 2.0, 3.7, "#F2F2F2", "#888",
          "Amino-acid<br>sequence")
    block(2.7, 4.3, 4.6, 5.3, "#E8F0FB", "#5B7FBF",
          "ESM2-150M<br><span style='font-size:10px'>frozen, fp16</span>")
    block(2.7, 1.7, 4.6, 2.7, "#E8F0FB", "#5B7FBF",
          "AlphaFoldDB<br><span style='font-size:10px'>predicted PDB</span>")
    block(5.0, 4.3, 7.4, 5.3, "#F2F2F2", "#888",
          "Per-residue X<br><span style='font-size:10px'>L × 640</span>")
    block(5.0, 1.7, 8.4, 2.7, "#F2F2F2", "#888",
          "Edge sets<br><span style='font-size:10px'>seq + contact <8Å</span>")
    block(9.0, 3.0, 10.5, 4.0, "#FFE9CF", "#D08F3C",
          "GATv2 × L<br><span style='font-size:10px'>h=64, H=8, L=3</span>")
    block(10.7, 3.0, 11.7, 4.0, "#FFE9CF", "#D08F3C",
          "Mean+attn<br>pool")
    block(11.9, 3.0, 12.5, 4.0, "#FFE9CF", "#D08F3C",
          "MLP")
    block(12.8, 3.0, 13.8, 4.0, "#FCE3E3", "#C45A5A",
          "σ(z) ∈ [0,1]<sup>10</sup>")
    block(8.7, 0.4, 12.4, 1.4, "#E5F5E5", "#5BAA5B",
          "GNNExplainer + raw GATv2 attention<br><span style='font-size:10px'>per-residue node importance</span>",
          italic=True)

    # arrows
    arrow(2.0, 3.2, 2.7, 4.8)   # seq -> esm
    arrow(2.0, 3.2, 2.7, 2.2)   # seq -> afdb
    arrow(4.6, 4.8, 5.0, 4.8)   # esm -> emb
    arrow(4.6, 2.2, 5.0, 2.2)   # afdb -> graph
    arrow(7.4, 4.8, 9.0, 3.7)   # emb -> gat
    arrow(8.4, 2.2, 9.0, 3.3)   # graph -> gat
    arrow(10.5, 3.5, 10.7, 3.5)  # gat -> pool
    arrow(11.7, 3.5, 11.9, 3.5)  # pool -> mlp
    arrow(12.5, 3.5, 12.8, 3.5)  # mlp -> logit
    arrow(9.7, 3.0, 10.0, 1.4, color="#5BAA5B")  # gat -> explain (dashed conceptually)

    fig.update_xaxes(visible=False, range=[0, 14])
    fig.update_yaxes(visible=False, range=[0.1, 5.7])
    fig.update_layout(**base_layout(margin=dict(l=10, r=10, t=10, b=10)))
    save(fig, "fig1_pipeline.pdf", w=900, h=380)


# =========================================================================
# FIG 2: ablation 3-panel bar chart
# =========================================================================
def fig2_ablation():
    rows = ["Classical RF", "ESM-MLP", "GAT-seq", "GAT-cont", "GAT-both (tuned)"]
    metrics = {
        "Macro ROC-AUC": [0.812, 0.913, 0.923, 0.925, 0.921],
        "Macro PR-AUC":  [0.376, 0.608, 0.644, 0.653, 0.673],
        "Macro F1 (τ=0.5)": [0.199, 0.497, 0.563, 0.561, 0.587],
    }
    colors = [PAL["classic"], PAL["esmmlp"], PAL["gatseq"], PAL["gatcon"], PAL["gatboth"]]

    fig = make_subplots(rows=1, cols=3, subplot_titles=list(metrics.keys()),
                        horizontal_spacing=0.09)
    for i, (mname, vals) in enumerate(metrics.items(), start=1):
        fig.add_trace(go.Bar(
            x=rows, y=vals, marker=dict(color=colors, line=dict(color="#222", width=0.6)),
            text=[f"{v:.3f}" for v in vals], textposition="outside",
            textfont=dict(size=10), cliponaxis=False, showlegend=False,
        ), row=1, col=i)
        fig.update_yaxes(range=[0, 1.0], gridcolor="#EEE", row=1, col=i,
                         tickfont=dict(size=10))
        fig.update_xaxes(tickangle=-25, tickfont=dict(size=10), row=1, col=i)

    for ann in fig.layout.annotations:
        ann.font = dict(family=FONT["family"], size=12, color="#222")

    fig.update_layout(**base_layout(margin=dict(l=50, r=20, t=44, b=80)))
    save(fig, "fig2_ablation.pdf", w=900, h=380)


# =========================================================================
# FIG 3: per-compartment ROC-AUC heatmap
# =========================================================================
def fig3_heatmap():
    compartments = ["Cytoplasm", "Nucleus", "Extracellular", "Cell mem.",
                    "Mitochondrion", "Plastid", "ER", "Lysosome",
                    "Golgi", "Peroxisome"]
    rows = ["LogReg", "RF", "MLP", "ESM-MLP", "GAT-seq", "GAT-cont", "<b>GAT-both</b>"]
    z = [
        [0.6976, 0.7651, 0.8374, 0.7496, 0.7989, 0.8786, 0.7968, 0.7221, 0.6831, 0.8531],
        [0.7470, 0.8120, 0.9037, 0.8030, 0.8374, 0.8850, 0.8184, 0.7495, 0.6799, 0.8867],
        [0.7298, 0.8109, 0.8940, 0.7759, 0.8263, 0.8741, 0.7973, 0.7430, 0.6625, 0.8691],
        [0.8446, 0.9056, 0.9789, 0.9160, 0.9450, 0.9877, 0.8970, 0.8560, 0.8324, 0.9712],
        [0.8684, 0.9206, 0.9828, 0.9252, 0.9561, 0.9877, 0.9113, 0.8663, 0.8444, 0.9633],
        [0.8697, 0.9220, 0.9842, 0.9250, 0.9556, 0.9885, 0.9111, 0.8755, 0.8444, 0.9699],
        [0.8700, 0.9211, 0.9834, 0.9226, 0.9495, 0.9885, 0.9094, 0.8757, 0.8305, 0.9601],
    ]
    text = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=compartments, y=rows, text=text, texttemplate="%{text}",
        textfont=dict(size=10, family=FONT["family"]),
        zmin=0.65, zmax=1.0,
        colorscale=[[0.0, "#3B4CC0"], [0.4, "#A0BDE0"], [0.55, "#DDDDDD"],
                    [0.7, "#F2A582"], [1.0, "#B40426"]],
        colorbar=dict(title=dict(text="ROC-AUC", side="right", font=dict(size=11)),
                      tickfont=dict(size=10), len=0.85, thickness=12,
                      tickvals=[0.7, 0.8, 0.9, 1.0]),
        hovertemplate="model %{y}<br>compartment %{x}<br>ROC-AUC %{z:.3f}<extra></extra>",
    ))
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=11), side="bottom")
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
    fig.update_layout(**base_layout(margin=dict(l=80, r=40, t=20, b=90)))
    save(fig, "fig3_heatmap.pdf", w=820, h=420)


# =========================================================================
# FIG 4: random-search sweep bar chart
# =========================================================================
def fig4_sweep():
    df = pd.read_csv(RUNS_GNN / "sweep" / "configs.csv")
    chosen = json.loads((RUNS_GNN / "sweep" / "chosen_config.json").read_text())
    chosen_id = int(chosen.get("config_id", -1))
    colors = [PAL["gatboth"] if int(c) == chosen_id else PAL["neutral"]
              for c in df["config_id"]]
    labels = [f"{int(r.layers)}L/h{int(r.hidden)}/H{int(r.heads)}/p{r.dropout}/lr{r.lr:.0e}"
              for r in df.itertuples()]

    fig = go.Figure(go.Bar(
        x=[str(c) for c in df["config_id"]], y=df["val_roc_macro"],
        marker=dict(color=colors, line=dict(color="#222", width=0.6)),
        text=[f"{v:.4f}" for v in df["val_roc_macro"]],
        textposition="outside", textfont=dict(size=9),
        cliponaxis=False, customdata=labels,
        hovertemplate="cfg %{x}<br>%{customdata}<br>val ROC %{y:.4f}<extra></extra>",
    ))
    fig.update_yaxes(range=[0.895, 0.915], gridcolor="#EEE",
                     tickformat=".3f", title="Macro val ROC-AUC",
                     tickfont=dict(size=11))
    fig.update_xaxes(title="Configuration index", tickfont=dict(size=11))

    # legend-ish annotation
    fig.add_annotation(x=0.99, y=1.05, xref="paper", yref="paper",
                       text=f"<span style='color:{PAL['gatboth']}'><b>■</b></span> chosen "
                            f"(L=3, h=64, H=8, p=0, lr=1e-3)",
                       showarrow=False, xanchor="right", yanchor="bottom",
                       font=dict(size=11))

    fig.update_layout(**base_layout(margin=dict(l=60, r=20, t=50, b=50)))
    save(fig, "fig4_sweep.pdf", w=720, h=380)


# =========================================================================
# FIG 5: GNNExplainer motif-overlap bars
# =========================================================================
def fig5_motifs():
    labels = ["Peroxisome<br><span style='font-size:10px;color:#777'>PTS1, n=35</span>",
              "ER<br><span style='font-size:10px;color:#777'>KDEL, n=50</span>",
              "Mitochondrion<br><span style='font-size:10px;color:#777'>N-term, n=50</span>",
              "Plastid<br><span style='font-size:10px;color:#777'>N-term, n=50</span>"]
    rates = [0.457, 0.000, 0.300, 0.000]
    colors = [PAL["hit"] if r > 0 else PAL["miss"] for r in rates]

    fig = go.Figure(go.Bar(
        x=labels, y=rates, marker=dict(color=colors, line=dict(color="#222", width=0.6)),
        text=[f"{r*100:.1f}%" if r > 0 else "0%" for r in rates],
        textposition="outside", textfont=dict(size=11),
        cliponaxis=False,
    ))
    fig.update_yaxes(range=[0, 0.55], tickformat=".0%", gridcolor="#EEE",
                     title="Importance–motif overlap rate",
                     tickfont=dict(size=11))
    fig.update_xaxes(tickfont=dict(size=11))
    fig.update_layout(**base_layout(margin=dict(l=70, r=20, t=20, b=70)))
    save(fig, "fig5_motifs.pdf", w=720, h=360)


# =========================================================================
# FIG 6: ROC + PR overlay curves for two illustrative compartments
# =========================================================================
def fig6_curves():
    # Read curves from each ablation row's test_curves.npz
    sources = {
        "RF (classical)": (RUNS_CLASSICAL / "curves.npz",       "rf",            PAL["classic"], "dot"),
        "ESM-MLP":        (RUNS_GNN / "ablation/mlp_pool/test_curves.npz", "mlp_pool_both", PAL["esmmlp"], "solid"),
        "GAT-seq":        (RUNS_GNN / "ablation/gat_seq/test_curves.npz",  "gat_sequence",  PAL["gatseq"], "solid"),
        "GAT-cont":       (RUNS_GNN / "ablation/gat_contact/test_curves.npz", "gat_contact", PAL["gatcon"], "solid"),
        "GAT-both":       (RUNS_GNN / "headline/test_curves.npz",          "gat_both",       PAL["gatboth"], "solid"),
    }

    def load(path: Path, model_key: str, label: str):
        d = np.load(path)
        yt = d.get(f"{model_key}__{label}__y_true")
        ys = d.get(f"{model_key}__{label}__y_score")
        return yt, ys

    compartments = [("Extracellular", "Extracellular"),
                    ("Peroxisome",   "Peroxisome")]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"ROC — {compartments[0][0]}",
                        f"PR — {compartments[0][0]}",
                        f"ROC — {compartments[1][0]}",
                        f"PR — {compartments[1][0]}"],
        horizontal_spacing=0.10, vertical_spacing=0.16,
    )

    for r_idx, (display, lkey) in enumerate(compartments, start=1):
        # diagonal reference for ROC
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(color="#BBB", width=1, dash="dash"),
                                 showlegend=False, hoverinfo="skip"),
                      row=r_idx, col=1)
        for legend_name, (path, model_key, color, dash) in sources.items():
            yt, ys = load(path, model_key, lkey)
            if yt is None or ys is None or len(np.unique(yt)) < 2:
                continue
            fpr, tpr, _ = roc_curve(yt, ys)
            prec, rec, _ = precision_recall_curve(yt, ys)
            lw = 2.4 if "GAT-both" in legend_name else 1.5
            show_legend = (r_idx == 1)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=legend_name,
                                     line=dict(color=color, width=lw, dash=dash),
                                     legendgroup=legend_name, showlegend=show_legend),
                          row=r_idx, col=1)
            fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=legend_name,
                                     line=dict(color=color, width=lw, dash=dash),
                                     legendgroup=legend_name, showlegend=False),
                          row=r_idx, col=2)

        fig.update_xaxes(title="FPR" if r_idx == 2 else "", row=r_idx, col=1,
                         range=[0, 1], gridcolor="#EEE", tickfont=dict(size=10))
        fig.update_yaxes(title="TPR", row=r_idx, col=1,
                         range=[0, 1.02], gridcolor="#EEE", tickfont=dict(size=10))
        fig.update_xaxes(title="Recall" if r_idx == 2 else "", row=r_idx, col=2,
                         range=[0, 1], gridcolor="#EEE", tickfont=dict(size=10))
        fig.update_yaxes(title="Precision", row=r_idx, col=2,
                         range=[0, 1.02], gridcolor="#EEE", tickfont=dict(size=10))

    for ann in fig.layout.annotations:
        ann.font = dict(family=FONT["family"], size=11, color="#222")

    fig.update_layout(**base_layout(
        margin=dict(l=60, r=20, t=50, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12,
                    xanchor="center", x=0.5, font=dict(size=10)),
    ))
    save(fig, "fig6_curves.pdf", w=820, h=620)


# =========================================================================
def main():
    print(f"writing figures to {OUT}")
    fig1_pipeline()
    fig2_ablation()
    fig3_heatmap()
    fig4_sweep()
    fig5_motifs()
    fig6_curves()
    print("done.")


if __name__ == "__main__":
    main()
