"""Generate EEGConformer architecture diagram for docs/figures/."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Always resolve paths relative to this file's location
_HERE = os.path.dirname(os.path.abspath(__file__))

# ── palette ───────────────────────────────────────────────────────────────────
C_INPUT    = "#E8F4FD"
C_PATCH    = "#D6EAF8"
C_POSENC   = "#FEF9E7"
C_ATTN     = "#D5F5E3"
C_FFN      = "#FDEDEC"
C_NORM     = "#E8DAEF"
C_HEAD     = "#FDEBD0"
C_OUT      = "#F2F3F4"
C_GRP_P_BG = "#EBF5FB"
C_GRP_T_BG = "#EAFAF1"
C_BDR      = "#2C3E50"
C_ARR      = "#566573"
C_GRP_P    = "#2E86C1"
C_GRP_T    = "#1E8449"

FW, FH = 10.0, 12.0
BX, BW = 1.6, 6.0          # box: left x, width
CX     = BX + BW / 2       # centre x
SX     = BX + BW + 0.26    # shape-label x (right of boxes)
GAP    = 0.30               # gap between major sections
IGAP   = 0.16               # gap between sub-boxes inside a group

fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")


# ── helpers ───────────────────────────────────────────────────────────────────

def fbox(y_top, h, label, sub="", color=C_INPUT, zorder=2):
    """Draw a rounded box. Returns (y_top, y_bot)."""
    y_bot = y_top - h
    ax.add_patch(FancyBboxPatch(
        (BX, y_bot), BW, h,
        boxstyle="round,pad=0.09",
        facecolor=color, edgecolor=C_BDR, linewidth=0.9, zorder=zorder,
    ))
    cy = y_bot + h / 2
    if sub:
        ax.text(CX, cy + 0.13, label,
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="#111111", zorder=zorder + 1)
        ax.text(CX, cy - 0.15, sub,
                ha="center", va="center", fontsize=7.5,
                color="#444444", zorder=zorder + 1)
    else:
        ax.text(CX, cy, label,
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="#111111", zorder=zorder + 1)
    return y_top, y_bot


def group_bg(y_top, y_bot, label, bg_color, bdr_color):
    """Dashed group background with rotated side label."""
    pad = 0.20
    ax.add_patch(FancyBboxPatch(
        (BX - pad, y_bot - pad), BW + 2 * pad, (y_top - y_bot) + 2 * pad,
        boxstyle="round,pad=0.12",
        facecolor=bg_color, edgecolor=bdr_color,
        linewidth=1.4, linestyle="--", zorder=1,
    ))
    ax.text(BX - pad - 0.16, (y_top + y_bot) / 2, label,
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color=bdr_color, rotation=90, zorder=3)


def arr(y_from, y_to):
    """Arrow from y_from downward to y_to."""
    ax.annotate(
        "", xy=(CX, y_to), xytext=(CX, y_from),
        arrowprops=dict(arrowstyle="-|>", color=C_ARR, lw=1.3, mutation_scale=14),
        zorder=5,
    )


def sl(y, text):
    """Shape label to the right of boxes."""
    ax.text(SX, y, text, ha="left", va="center",
            fontsize=7.5, color="#7F8C8D", style="italic", zorder=3)


# ── layout (top → bottom) ────────────────────────────────────────────────────

Y = FH - 0.55
ax.text(CX, Y, "EEGConformer — Architecture",
        ha="center", va="center", fontsize=12, fontweight="bold", color=C_BDR)
Y -= 0.62

sections = []   # (top, bot) per major section; used for main-flow arrows

# Input
t, b = fbox(Y, 0.58, "Input EEG Epoch",
            "(B, 56 channels, 260 samples)  ·  float32", C_INPUT)
sl((t + b) / 2, "(B, 56, 260)")
sections.append((t, b));  Y = b - GAP

# ── Patch Embedding ───────────────────────────────────────────────────────────
pe_top = Y

t1, b1 = fbox(Y, 0.65, "Temporal Conv2d  (1 → 64,  kernel 1×25)",
              "BatchNorm  ·  ELU  ·  AvgPool(kernel=75, stride=15)  →  T′=13",
              C_PATCH)
sl((t1 + b1) / 2, "(B, 64, 56, 13)")
Y = b1 - IGAP

t2, b2 = fbox(Y, 0.65, "Spatial DepthwiseConv2d  (64 → 64,  kernel 56×1)",
              "BatchNorm  ·  ELU  ·  squeeze & permute",
              C_PATCH)
sl((t2 + b2) / 2, "(B, 13, 64)")
pe_bot = b2

group_bg(pe_top, pe_bot, "Patch Embedding", C_GRP_P_BG, C_GRP_P)
arr(b1, t2)                             # temporal → spatial (internal arrow)
sections.append((pe_top, pe_bot));  Y = pe_bot - GAP

# Positional Encoding
t, b = fbox(Y, 0.55, "+  Learnable Positional Encoding",
            "parameter  (1, 13, 64)  —  added element-wise", C_POSENC)
sl((t + b) / 2, "(B, 13, 64)")
sections.append((t, b));  Y = b - GAP

# ── Transformer Block  ×4 ─────────────────────────────────────────────────────
tr_top = Y

t1, b1 = fbox(Y, 0.62, "Multi-Head Self-Attention  (8 heads,  dim=64)",
              "Residual connection  +  LayerNorm", C_ATTN)
sl((t1 + b1) / 2, "(B, 13, 64)")
Y = b1 - IGAP

t2, b2 = fbox(Y, 0.62, "Feed-Forward Network  (64 → 256 → 64,  GELU)",
              "Residual connection  +  LayerNorm", C_FFN)
sl((t2 + b2) / 2, "(B, 13, 64)")
tr_bot = b2

group_bg(tr_top, tr_bot, "Transformer  ×4", C_GRP_T_BG, C_GRP_T)
arr(b1, t2)                             # MHA → FFN (internal arrow)
sections.append((tr_top, tr_bot));  Y = tr_bot - GAP

# ── Classifier Head ───────────────────────────────────────────────────────────
t, b = fbox(Y, 0.50, "LayerNorm  (dim=64)", "", C_NORM)
sl((t + b) / 2, "(B, 13, 64)")
sections.append((t, b));  Y = b - 0.22

t, b = fbox(Y, 0.50, "Flatten", "13 × 64 = 832", C_HEAD)
sl((t + b) / 2, "(B, 832)")
sections.append((t, b));  Y = b - 0.22

t, b = fbox(Y, 0.58, "Linear  (832 → 256)  +  GELU  +  Dropout(0.5)", "", C_HEAD)
sl((t + b) / 2, "(B, 256)")
sections.append((t, b));  Y = b - 0.22

t, b = fbox(Y, 0.50, "Linear  (256 → 1)", "raw logit", C_HEAD)
sl((t + b) / 2, "(B, 1)")
sections.append((t, b));  Y = b - GAP

# Output
t, b = fbox(Y, 0.55, "Output",
            "sigmoid(logit)  →  P(ErrP)  ∈  [0, 1]", C_OUT)
sl((t + b) / 2, "float  ∈  [0, 1]")
sections.append((t, b))

# ── Main-flow arrows ──────────────────────────────────────────────────────────
for i in range(len(sections) - 1):
    arr(sections[i][1], sections[i + 1][0])

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(_HERE, "figures", "architecture.png")
plt.tight_layout(pad=0.3)
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved {out}")
