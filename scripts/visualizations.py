#Evan New
#Create visuals to compare central vs wide progressions

#import necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mplsoccer import Pitch

#define constants, load data, and create a place to output visuals
FIG_DIR  = "figures/"
PROG_CSV = "events/progressions_checkpoint.csv"
PV_CSV   = "events/pv_table.csv"
X_MAX    = 120
Y_MAX    = 80
CENTRAL_COLOR = "blue"
WIDE_COLOR    = "red"
progressions = pd.read_csv(PROG_CSV)
pv_df        = pd.read_csv(PV_CSV)

#create bar chart that compares overall outcomes of central vs. wide progression
outcomes = progressions.groupby("DOMINANT_ZONE")[["SHOT", "XG", "TURNOVER"]].mean()
metrics = ["SHOT", "XG", "TURNOVER"]
metric_labels = ["Shot Rate", "Average xG", "Turnover Rate"]
central_vals = [outcomes.loc["central", m] for m in metrics]
wide_vals = [outcomes.loc["wide", m] for m in metrics]
x  = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
bars_c = ax.bar(x - width / 2, central_vals, width, label="Central", color=CENTRAL_COLOR)
bars_w = ax.bar(x + width / 2, wide_vals, width, label="Wide", color=WIDE_COLOR)
for bar in bars_c:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars_w:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=12)
ax.set_ylabel("Rate", fontsize=12)
ax.set_title("Overall Outcome Rates by Zone", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.set_ylim(0, max(max(central_vals), max(wide_vals)) * 1.25)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}overall_outcomes.png", dpi=150, bbox_inches="tight")
plt.close()

#create bar chart that compares progression value of central vs. wide progressions by field third
pv_third = (pv_df.groupby(["THIRD", "ZONE"]).apply(lambda g: np.average(g["PV"], weights=g["COUNT"])).reset_index(name="PV"))
thirds = ["defensive_third", "middle_third", "final_third"]
third_labels = ["Defensive Third", "Middle Third", "Final Third"]
central_pv = [pv_third[(pv_third["THIRD"] == t) & (pv_third["ZONE"] == "central")]["PV"].values[0] for t in thirds]
wide_pv = [pv_third[(pv_third["THIRD"] == t) & (pv_third["ZONE"] == "wide")]["PV"].values[0] for t in thirds]
x = np.arange(len(thirds))
fig, ax = plt.subplots(figsize=(8, 5))
bars_c = ax.bar(x - width / 2, central_pv, width, label="Central", color=CENTRAL_COLOR)
bars_w = ax.bar(x + width / 2, wide_pv, width, label="Wide", color=WIDE_COLOR)
for bar in bars_c:
    yval = bar.get_height()
    offset = 0.001 if yval >= 0 else -0.003
    ax.text(bar.get_x() + bar.get_width() / 2, yval + offset, f"{yval:.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars_w:
    yval = bar.get_height()
    offset = 0.001 if yval >= 0 else -0.003
    ax.text(bar.get_x() + bar.get_width() / 2, yval + offset, f"{yval:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(third_labels, fontsize=12)
ax.set_ylabel("Progression Value (PV)", fontsize=12)
ax.set_title("Progression Value by Field Third", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}PV_by_third.png", dpi=150, bbox_inches="tight")
plt.close()

#create bar chart that compares progression value of central vs. wide progressions by score state
pv_score = (pv_df.groupby(["SCORE_STATE", "ZONE"]).apply(lambda g: np.average(g["PV"], weights=g["COUNT"])).reset_index(name="PV"))
states = ["losing", "tied", "winning"]
state_labels = ["Losing", "Tied", "Winning"]
central_pv_s = [pv_score[(pv_score["SCORE_STATE"] == s) & (pv_score["ZONE"] == "central")]["PV"].values[0] for s in states]
wide_pv_s    = [pv_score[(pv_score["SCORE_STATE"] == s) & (pv_score["ZONE"] == "wide")]["PV"].values[0] for s in states]
x = np.arange(len(states))
fig, ax = plt.subplots(figsize=(8, 5))
bars_c = ax.bar(x - width / 2, central_pv_s, width, label="Central", color=CENTRAL_COLOR)
bars_w = ax.bar(x + width / 2, wide_pv_s, width, label="Wide", color=WIDE_COLOR)
for bar in bars_c:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.001, f"{yval:.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars_w:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.001, f"{yval:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(state_labels, fontsize=12)
ax.set_ylabel("Progression Value (PV)", fontsize=12)
ax.set_title("Progression Value by Score State", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}PV_by_score.png", dpi=150, bbox_inches="tight")
plt.close()

#bar chart that compares progression value of central vs. wide progressions by defensive shape and field third
pv_by_def = (pv_df.groupby(["THIRD", "DEF_WIDTH_CATEGORY", "ZONE"]) .apply(lambda g: np.average(g["PV"], weights=g["COUNT"])) .reset_index(name="PV"))

thirds_def = ["defensive_third", "middle_third", "final_third"]
third_labels_def = ["Defensive Third", "Middle Third", "Final Third"]
def_cats = ["compact", "stretched"]
def_labels = ["Compact", "Stretched"]
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
x  = np.arange(len(def_cats))
width = 0.35
for ax, third, title in zip(axes, thirds_def, third_labels_def):
    central_vals = [pv_by_def[(pv_by_def["THIRD"] == third) & (pv_by_def["DEF_WIDTH_CATEGORY"] == d) & (pv_by_def["ZONE"] == "central")]["PV"].values[0]for d in def_cats]
    wide_vals = [pv_by_def[(pv_by_def["THIRD"] == third) & (pv_by_def["DEF_WIDTH_CATEGORY"] == d) & (pv_by_def["ZONE"] == "wide")]["PV"].values[0]for d in def_cats]
    bars_c = ax.bar(x - width / 2, central_vals, width, label="Central", color=CENTRAL_COLOR)
    bars_w = ax.bar(x + width / 2, wide_vals, width, label="Wide", color=WIDE_COLOR)
    for bar in bars_c:
        yval = bar.get_height()
        offset = 0.001 if yval >= 0 else -0.004
        ax.text(bar.get_x() + bar.get_width() / 2, yval + offset, f"{yval:.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_w:
        yval = bar.get_height()
        offset = 0.001 if yval >= 0 else -0.004
        ax.text(bar.get_x() + bar.get_width() / 2, yval + offset,f"{yval:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(def_labels, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Progression Value (PV)", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)
fig.suptitle("Progression Value by Defensive Shape and Field Third", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}PV_by_Shape.png", dpi=150, bbox_inches="tight")
plt.close()

#create diagram of progression start positions
progs = progressions.copy()
progs["ATT_X"] = progs.apply(lambda r: r["START_X"] if r["TEAM_ID"] == 0 else X_MAX - r["START_X"], axis=1)
progs["ATT_Y"] = progs["START_Y"]
central_progs = progs[progs["DOMINANT_ZONE"] == "central"]
wide_progs = progs[progs["DOMINANT_ZONE"] == "wide"]
pitch = Pitch(pitch_type="custom", pitch_length=X_MAX, pitch_width=Y_MAX, pitch_color="#1f6028", line_color="white", linewidth=1.5)
fig, ax = pitch.draw(figsize=(12, 7))
ax.scatter(wide_progs["ATT_X"], wide_progs["ATT_Y"],c=WIDE_COLOR, alpha=0.25, s=8, label="Wide", zorder=3)
ax.scatter(central_progs["ATT_X"], central_progs["ATT_Y"], c=CENTRAL_COLOR, alpha=0.25, s=8, label="Central", zorder=4)
central_patch = mpatches.Patch(color=CENTRAL_COLOR, label=f"Central (n={len(central_progs):,})")
wide_patch = mpatches.Patch(color=WIDE_COLOR, label=f"Wide (n={len(wide_progs):,})")
ax.legend(handles=[central_patch, wide_patch], fontsize=11, loc="upper left", framealpha=0.8)
ax.set_title("Progression Start Positions by Zone (attack from left to right)", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}pitch_diagram.png", dpi=150, bbox_inches="tight")
plt.close()
